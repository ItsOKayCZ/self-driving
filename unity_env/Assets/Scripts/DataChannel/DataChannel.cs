using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class DataChannel : SideChannel
{
    private static Dictionary<string, int> intParameters = new Dictionary<string, int>();
    private static Dictionary<string, Color> colorParameters = new Dictionary<string, Color>();
    private static Dictionary<string, float> floatParameters = new Dictionary<string, float>();

    public static int getParameter(string key, int defaultValue)
    {
        if (intParameters.ContainsKey(key))
            return intParameters[key];
        return defaultValue;
    }

    public static float getParameter(string key, float defaultValue)
    {
        if (floatParameters.ContainsKey(key))
            return floatParameters[key];
        return defaultValue;
    }

    public static Color getParemeter(string key, Color defaultValue)
    {
        if (colorParameters.ContainsKey(key))
            return colorParameters[key];
        return defaultValue;
    }

    public DataChannel()
    {
        ChannelId = new Guid("9bc23f51-e0e8-450c-b3c5-e4d2032151ec");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        string receivedString = msg.ReadString();

        string[] segments = receivedString.Split('|');
        switch (segments[0])
        {
            case "int":
                intParameters[segments[1]] = int.Parse(segments[2]);
                break;

            case "color":
                string[] rgb = segments[2].Split(',');
                colorParameters[segments[1]] = new Color(
                    float.Parse(rgb[0]) / 255f,
                    float.Parse(rgb[1]) / 255f,
                    float.Parse(rgb[2]) / 255f
                );
                break;

            case "float":
                floatParameters[segments[1]] = float.Parse(segments[2]);
                break;

            default:
                Debug.LogError("Didn't recognize data type of message");
                break;
        }
    }
}
