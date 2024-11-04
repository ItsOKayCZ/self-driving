using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class RegisterDataChannel : MonoBehaviour
{
    DataChannel dataChannel;

    public void Awake()
    {
        dataChannel = new DataChannel();

        SideChannelManager.RegisterSideChannel(dataChannel);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(dataChannel);
        }
    }
}
