using UnityEngine;

namespace Unity.MLAgents.Areas
{
	[DefaultExecutionOrder(-5)]
	public class TrainingReplicator : MonoBehaviour
	{

		public GameObject baseArea;
		public int numAreas = 1;
		public float margin = 20;

		public void Awake()
		{
			if (Academy.Instance.IsCommunicatorOn)
				numAreas = Academy.Instance.NumAreas;
		}

		void OnEnable()
		{
			AddAreas();
		}

		private void AddAreas()
		{
			for(int i = 0; i < numAreas; i++)
			{
				if (i == 0)
					continue;

				Vector3 pos = Vector3.up * margin * i;
				Instantiate(baseArea, pos, Quaternion.identity);
			}
		}
	}

}
