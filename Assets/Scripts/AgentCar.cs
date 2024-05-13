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
	const int k_Back = 1;
	const int k_Left = 2;
	const int k_Right = 3;

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
		pauseLearning = false;

		currentCheckpoint = 0;

		transform.position = transform.parent.position;
		transform.rotation = Quaternion.identity;

        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
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
		bool goForward = actions.DiscreteActions[k_Forward] == 1;
		// bool goForward = true;
		bool goBack = actions.DiscreteActions[k_Back] == 1;
		// bool goBack = false;
		bool turnLeft = actions.DiscreteActions[k_Left] == 1;
		bool turnRight = actions.DiscreteActions[k_Right] == 1;

		carController.Movement(true, goForward, goBack, turnLeft, turnRight);
	}

	public override void OnActionReceived(ActionBuffers actions)
	{
		if (pauseLearning)
			return;

		float distanceToCheckpoint = calcDistanceToNextCheckpoint();
		if(distanceToCheckpoint != -1 && distanceToCheckpoint < 2.5f)
		{
			AddReward(5f);
			currentCheckpoint++;
			trackGenerator.UpdateTrack(currentCheckpoint);
		}

		if (carController.getAmountOfWheelsOnRoad() <= 2)
		{
			Debug.Log("Tire on terrain. Resetting");
			SetReward(-10f);
			EndEpisode();
		}

		// SetReward(carController.getAmountOfWheelsOnRoad() * 0.0001f);
		// SetReward(4 - carController.getAmountOfWheelsOnRoad() * -0.1f);

		if(carController.carSpeed > 2f)
		{
			float reward = getDrivenDistance();
			// Debug.Log(reward);
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

		discreteActionsOut[k_Forward] = Input.GetKey(KeyCode.W) ? 1 : 0;
		discreteActionsOut[k_Back] = Input.GetKey(KeyCode.S) ? 1 : 0;
		discreteActionsOut[k_Left] = Input.GetKey(KeyCode.A) ? 1 : 0;
		discreteActionsOut[k_Right] = Input.GetKey(KeyCode.D) ? 1 : 0;
	}
}