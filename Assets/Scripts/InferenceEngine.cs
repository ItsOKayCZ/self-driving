using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;

public class InferenceEngine : MonoBehaviour
{
    // TODO: Use sentis instead of barracuda
    public ModelAsset modelAsset;
    private Model runtimeModel;

    public RenderTexture renderTexture;
    public PrometeoCarController carController;

    private IWorker worker;
    
    // Start is called before the first frame update
    void Start()
    {
        if (modelAsset == null)
        {
           enabled = false;
           return;
        }

		runtimeModel = ModelLoader.Load(modelAsset);

        worker = WorkerFactory.CreateWorker(BackendType.CPU, runtimeModel, verbose: false);
        List<Model.Input> inputs = runtimeModel.inputs;

        foreach(var input in inputs){
            Debug.Log(input.name);
            Debug.Log(input.shape);
        }
    }

    private void Update()
    {
        TensorFloat vision = TextureConverter.ToTensor(renderTexture);

        float[] steering = new float[]{ carController.steeringAxis };
        TensorShape shape = new TensorShape(1, 1);
        TensorFloat nonVision = new TensorFloat(shape, steering);

        Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>(){
            {"vis_obs", vision},
            {"nonvis_obs", nonVision}
        };

        worker.Execute(inputs);

        TensorFloat speedOutput = worker.PeekOutput("47") as TensorFloat;
        speedOutput.MakeReadable();
        float[] speed = speedOutput.ToReadOnlyArray();

        TensorFloat steerOutput = worker.PeekOutput("53") as TensorFloat;
        steerOutput.MakeReadable();
        float[] steer = steerOutput.ToReadOnlyArray();
        
        carController.Movement(true, speed[0], steer[0]);

        speedOutput.Dispose();
        steerOutput.Dispose();
    }
}
