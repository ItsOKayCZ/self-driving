using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

public class AgentCar : Agent
{
	public GameObject parentCheckpoint;
	List<GameObject> checkpoints
	{
		get {
			return trackGenerator.checkpoints;
		}
	}


	public int currentCheckpoint = 0;

	private bool pauseLearning = false;

	const int k_Forward = 0;
	const int k_Turn = 1;
	// const int k_Back = 1;
	// const int k_Left = 0;
	// const int k_Right = 1;

	public PrometeoCarController carController;
	// public CarController carController;
	public Rigidbody rBody;
	public TrackGenerator trackGenerator;

	public void Start()
	{
		carController.useControls = false;
	}

	public override void OnEpisodeBegin()
	{
		Debug.Log("New episode");

		pauseLearning = true;
		trackGenerator.ResetTrack();
		transform.position = transform.parent.position;
		transform.rotation = Quaternion.identity;

		rBody.velocity = Vector3.zero;
		rBody.angularVelocity = Vector3.zero;

		float brakeForce = 1000f;
		carController.frontLeftCollider.brakeTorque = brakeForce;
		carController.frontRightCollider.brakeTorque = brakeForce;
		carController.rearLeftCollider.brakeTorque = brakeForce;
		carController.rearRightCollider.brakeTorque = brakeForce;

		pauseLearning = false;

		currentCheckpoint = 0;
	}

	public override void CollectObservations(VectorSensor sensor)
	{
		// sensor.AddObservation(carController.carSpeed);
		// sensor.AddObservation(carController.currentSteerAngle);
		sensor.AddObservation(carController.steeringAxis);

		// sensor.AddObservation(calcDistanceToNextCheckpoint());
	}

	private GameObject getNextCheckpoint()
	{
		if(currentCheckpoint + 1 >= checkpoints.Count)
			return checkpoints[0];
		return checkpoints[currentCheckpoint + 1];
	}

	private float calcDistance(Vector3 pos1, Vector3 pos2)
	{
		return Vector2.Distance(
			new Vector2(pos1.x, pos1.z),
			new Vector2(pos2.x, pos2.z)
		);
	}

	private float calcDistanceToNextCheckpoint()
	{
		if(checkpoints.Count == 0)
			return -1;

		GameObject nextCheckpoint = getNextCheckpoint();
		if (nextCheckpoint == null)
			return -1;

		return calcDistance(nextCheckpoint.transform.position, transform.position);
	}

	private float getDrivenDistance()
	{
		float distance = 0f;

		for(int i = 0; i < currentCheckpoint; i++)
		{
			distance += calcDistance(
				checkpoints[i].transform.position,
				checkpoints[i + 1].transform.position
			);
		}

		distance += calcDistance(
			checkpoints[checkpoints.Count - (checkpoints.Count - currentCheckpoint)].transform.position,
			transform.position
		);

		return distance;
	}

	void TriggerAction(ActionBuffers actions)
	{
		float speedMult = actions.ContinuousActions[k_Forward];
		float turnMult = actions.ContinuousActions[k_Turn];

		carController.Movement(true, speedMult, turnMult);
	}

	public override void OnActionReceived(ActionBuffers actions)
	{
		if (pauseLearning)
			return;

		float distanceToCheckpoint = calcDistanceToNextCheckpoint();
		if(distanceToCheckpoint != -1 && distanceToCheckpoint < 10f)
		{
			AddReward(10f);
			currentCheckpoint++;
			trackGenerator.UpdateTrack(currentCheckpoint);
		}

		if (carController.getAmountOfWheelsOnRoad() <= 2)
		{
			Debug.Log("Tire on terrain. Resetting");
			SetReward(-5f);
			EndEpisode();
		}

		// SetReward(carController.getAmountOfWheelsOnRoad() * 0.0001f);
		// SetReward(4 - carController.getAmountOfWheelsOnRoad() * -0.1f);

		if(carController.carSpeed > 2f)
		{
			float rewardMultBySpeed = Mathf.Clamp(carController.carSpeed / carController.maxSpeed, 0f, 1f);
			float reward = getDrivenDistance() * rewardMultBySpeed;

			AddReward(reward);
		} else
		{
			AddReward(-5f);
		}

		TriggerAction(actions);
	}

	public override void Heuristic(in ActionBuffers actionsOut)
	{
		var discreteActionsOut = actionsOut.DiscreteActions;
		var continuousActionsOut = actionsOut.ContinuousActions;

		continuousActionsOut[k_Forward] = Mathf.Clamp(Input.GetAxisRaw("Vertical"), 0f, 1f);
		continuousActionsOut[k_Turn] = Input.GetAxisRaw("Horizontal");
	}
}