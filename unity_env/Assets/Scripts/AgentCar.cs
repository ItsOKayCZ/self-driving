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

    const int k_Forward = 0;
    const int k_Back = 1;
    const int k_Left = 2;
    const int k_Right = 3;

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
        var discreteActionsOut = actionsOut.DiscreteActions;

        discreteActionsOut[k_Forward] = Input.GetKey(KeyCode.W) ? 1 : 0;
        discreteActionsOut[k_Back] = Input.GetKey(KeyCode.S) ? 1 : 0;
        discreteActionsOut[k_Left] = Input.GetKey(KeyCode.A) ? 1 : 0;
        discreteActionsOut[k_Right] = Input.GetKey(KeyCode.D) ? 1 : 0;
    }
}
