using System;
using System.Collections.Generic;
using System.Net.Http.Headers;
using Dreamteck.Splines;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class AgentCar : Agent
{
    public GameObject parentCheckpoint;
    List<SplineComputer> checkpoints
    {
        get { return trackGenerator.checkpoints; }
    }

    public int currentCheckpoint = 0;

    private bool pauseLearning = false;

    const int k_Speed = 0;
    const int k_Steering = 1;

    public PrometeoCarController carController;

    public Camera carCamera;

    // public CarController carController;
    public Rigidbody rBody;
    public TrackGenerator trackGenerator;

    float deathPenalty = -10f;

    public void Start()
    {
        carController.useControls = false;

        deathPenalty = DataChannel.getParameter("deathPenalty", -10f);
    }

    public override void OnEpisodeBegin()
    {
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
        sensor.AddObservation(carController.steeringAxis);
        sensor.AddObservation(Mathf.Clamp(carController.carSpeed, 0f, 1f));
    }

    private float calcDistanceToCenter()
    {
        SplineSample splineSample = new SplineSample();
        checkpoints[currentCheckpoint].Project(transform.position, ref splineSample);

        float dist = Vector2.Distance(
            new Vector2(transform.position.x, transform.position.z),
            new Vector2(splineSample.position.x, splineSample.position.z)
        );
        float val = 1f - (dist / 6.34f);

        return val;
    }

    private bool isOverLastPoint()
    {
        int lastIndex = checkpoints[currentCheckpoint].pointCount - 1;
        Vector3 piecePos = trackGenerator.track[currentCheckpoint].go.transform.position;
        Vector3 endLinePos = checkpoints[currentCheckpoint].GetPoint(lastIndex).position;

        Vector2 startPos = new Vector2(piecePos.x, piecePos.z);
        Vector2 endPos = new Vector2(endLinePos.x, endLinePos.z);
        Vector2 agentPos = new Vector3(transform.position.x, transform.position.z);

        Vector2 pieceDir = endPos - startPos;
        Vector2 agentEndDir = endPos - agentPos;

        float dot = Vector2.Dot(pieceDir, agentEndDir);

        return dot <= 2f;
    }

    void TriggerAction(ActionBuffers actions)
    {
        float speed = actions.ContinuousActions[k_Speed];
        float steering = actions.ContinuousActions[k_Steering];

        carController.Movement(true, speed, steering);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (pauseLearning)
            return;

        SetReward(calcDistanceToCenter());

        if (isOverLastPoint())
        {
            currentCheckpoint++;
            trackGenerator.UpdateTrack(currentCheckpoint);
        }

        if (carController.getAmountOfWheelsOnRoad() <= 2)
        {
            SetReward(deathPenalty);
            EndEpisode();
        }

        AddReward((4 - carController.getAmountOfWheelsOnRoad()) * -1f);

        TriggerAction(actions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        continuousActionsOut[k_Speed] = 0;
        if (Input.GetKey(KeyCode.W))
            continuousActionsOut[k_Speed] += 1;

        if (Input.GetKey(KeyCode.S))
            continuousActionsOut[k_Speed] -= 1;

        continuousActionsOut[k_Steering] = 0;
        if (Input.GetKey(KeyCode.D))
            continuousActionsOut[k_Steering] += 1;

        if (Input.GetKey(KeyCode.A))
            continuousActionsOut[k_Steering] -= 1;
    }
}
