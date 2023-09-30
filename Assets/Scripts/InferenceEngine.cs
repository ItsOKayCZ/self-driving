using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class InferenceEngine : MonoBehaviour
{
    public NNModel modelAsset;
    private Model runtimeModel;

    public RenderTexture renderTexture;
    public PrometeoCarController carController;

    private IWorker worker;
    
    // Start is called before the first frame update
    void Start()
    {
        if (modelAsset == null)
        {
            this.enabled = false;
            return;
        }

        runtimeModel = ModelLoader.Load(modelAsset);

        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, runtimeModel);
    }

    private void Update()
    {
        var inputs = new Dictionary<string, Tensor>();
        inputs["vis_obs"] = new Tensor(renderTexture, 1);
        inputs["nonvis_obs"] = new Tensor(new TensorShape(1, 1));

        inputs["nonvis_obs"][0, 0] = carController.steeringAxis;

        worker.Execute(inputs);

        var actions = worker.PeekOutput("action");
        bool goForward = actions[0, 0, 0, 0] == 1;
        bool goBack = actions[0, 0, 0, 1] == 1;
        bool turnRight = actions[0, 0, 0, 3] == 1;
        bool turnLeft = actions[0, 0, 0, 2] == 1;
        Debug.Log(actions[0, 0, 0, 0]);
        Debug.Log(actions[0, 0, 0, 1]);
        Debug.Log(actions[0, 0, 0, 2]);
        Debug.Log(actions[0, 0, 0, 3]);
        Debug.Log("___________________________");

        // carController.Movement(true, goForward, goBack, turnLeft, turnRight);

        actions.Dispose();
    }
}
