using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Wheel : MonoBehaviour
{
    WheelCollider wCollider;

    public bool onRoad = true;

    private void Start()
    {
        wCollider = GetComponent<WheelCollider>();
    }

    private void Update()
    {
        WheelHit hit;
        if(wCollider.GetGroundHit(out hit))
        {
            // onRoad = (hit.collider.gameObject.name != "Terrain");
            onRoad = (hit.collider.gameObject.tag == "Road");
        }
    
        onRoad = onRoad && wCollider.isGrounded;
    }
}
