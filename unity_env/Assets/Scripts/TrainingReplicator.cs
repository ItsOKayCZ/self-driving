using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Areas
{
    [DefaultExecutionOrder(-5)]
    public class TrainingReplicator : MonoBehaviour
    {
        public GameObject baseArea;
        public int numAreas = 1;
        public float margin = 20;

        public enum RoadColor
        {
            Amazon,
            BlackWhite,
        };

        // Wide: 15
        // Slim: 10
        public int roadSize = 15;
        public RoadColor roadColor;
        public Color backgroundColor;

        public void Awake()
        {
            if (Academy.Instance.IsCommunicatorOn)
            {
                numAreas = Academy.Instance.NumAreas;
            }
        }

        void Start()
        {
            roadSize = DataChannel.getParameter("roadSize", 15);
            roadColor = (RoadColor)DataChannel.getParameter("roadColor", 0);
            backgroundColor = DataChannel.getParemeter(
                "backgroundColor",
                new Color(0, 0.819607843f, 0.529411765f)
            );

            ChangeCameraSettings();
            AddAreas();
        }

        private void ChangeCameraSettings()
        {
            GameObject.Find("Car camera").GetComponent<Camera>().backgroundColor = backgroundColor;
            GameObject.Find("Camera").GetComponent<Camera>().backgroundColor = backgroundColor;
            // if (roadColor == RoadColor.Amazon)
            // {
            //     GameObject.Find("Car camera").GetComponent<Camera>().backgroundColor = new Color(
            //         0,
            //         209f / 255f,
            //         135f / 255f
            //     );
            // }
            // else
            // {
            //     GameObject.Find("Car camera").GetComponent<Camera>().backgroundColor = new Color(
            //         1f,
            //         1f,
            //         1f
            //     );
            // }
        }

        private void AddAreas()
        {
            for (int i = 0; i < numAreas; i++)
            {
                if (i == 0)
                    continue;

                Vector3 pos = Vector3.up * margin * i;
                Instantiate(baseArea, pos, Quaternion.identity);
            }
        }
    }
}
