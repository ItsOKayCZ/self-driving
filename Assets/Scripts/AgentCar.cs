using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System;
using UnityEngine.Playables;
using Unity.Sentis.Layers;
using TMPro;
using System.ComponentModel;
using System.Data;
using UnityEngine.UIElements;
using Dreamteck.Splines;
using Unity.VisualScripting;
using System.Linq;

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
	private Vector2 previousPosition;
	private Vector2 checkpointDir;
	private float distanceMultiplier = 1f;

	private bool pauseLearning = false;

	const int k_Forward = 0;
	const int k_Turn = 1;
	// const int k_Back = 1;
	// const int k_Left = 0;
	// const int k_Right = 1;

	public PrometeoCarController carController;
	public Rigidbody rBody;
	public TrackGenerator trackGenerator;

	public void Start()
	{
		carController.useControls = false;

		var envParameters = Academy.Instance.EnvironmentParameters;
		distanceMultiplier = envParameters.GetWithDefault("distanceMultiplier", 1f);
	}

	private Vector2 convert(Vector3 v)
	{
		return new Vector2(v.x, v.z);
	}

	public override void OnEpisodeBegin()
	{
		Debug.Log("New episode");

		pauseLearning = true;
		trackGenerator.ResetTrack();
		transform.position = transform.parent.position;
		transform.rotation = Quaternion.identity;
		
		previousPosition = convert(transform.position);

		rBody.velocity = Vector3.zero;
		rBody.angularVelocity = Vector3.zero;

		float brakeForce = 1000f;
		carController.frontLeftCollider.brakeTorque = brakeForce;
		carController.frontRightCollider.brakeTorque = brakeForce;
		carController.rearLeftCollider.brakeTorque = brakeForce;
		carController.rearRightCollider.brakeTorque = brakeForce;

		pauseLearning = false;

		currentCheckpoint = 0;
		checkpointDir = convert(checkpoints[currentCheckpoint + 1].transform.position).normalized;
	}

	public override void CollectObservations(VectorSensor sensor)
	{
		sensor.AddObservation(carController.steeringAxis);
		sensor.AddObservation(Mathf.Clamp(carController.carSpeed / carController.maxSpeed, 0f, 1f));
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

	private float calcDistanceToNextNextCheckpoint()
	{
		if(checkpoints.Count == 0)
			return -1;

		GameObject checkpoint = checkpoints[currentCheckpoint + 2];

		return calcDistance(checkpoint.transform.position, transform.position);
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

	private float getDrivenDistanceRelative()
	{
		Vector2 currentPosition = convert(transform.position);
		Vector2 dir = currentPosition - previousPosition;
		float dist = dir.magnitude;

		previousPosition = convert(transform.position);

		if(Vector2.Dot(checkpointDir, dir) > 0)
			return Mathf.Clamp(carController.carSpeed / carController.maxSpeed, 0f, 1f);
		
		if(carController.carSpeed <= 1f)
			return -0.5f;
		return -1f;
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

		float reward = 0;

		float distanceToCheckpoint = calcDistanceToNextCheckpoint();
		if(distanceToCheckpoint != -1 && distanceToCheckpoint < 5f)
		{
			currentCheckpoint++;
			trackGenerator.UpdateTrack(currentCheckpoint);
			checkpointDir = convert(checkpoints[currentCheckpoint + 1].transform.position - checkpoints[currentCheckpoint].transform.position).normalized;
		}

		int amountOfWheelsOnRoad = carController.getAmountOfWheelsOnRoad();
		if(4 - amountOfWheelsOnRoad >= 1f)
		{
			Debug.Log("Tire on terrain. Resetting.");
			SetReward(-10f);
			EndEpisode();
		}
		reward += (4 - amountOfWheelsOnRoad) * -2f;

		if(carController.carSpeed < 1f)
			reward = -5f;
		else {
			reward += getDrivenDistance();
			// reward += Mathf.Clamp(carController.carSpeed / carController.maxSpeed, 0f, 1f) * 2f;
			// reward *= Mathf.Pow(2f - trackGenerator.calcDistanceToMiddleOfRoad(currentCheckpoint, transform.position), 2f);
		}
		Debug.Log(reward);

		SetReward(reward);

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