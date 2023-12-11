using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CameraManager : MonoBehaviour
{
    public FreeFlyCamera freeFlyCamera;
    public Toggle toggleEl;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    public void ToggleCamera()
    {
        SetCamera();
    }

    private void SetCamera(){
        freeFlyCamera.enabled = !freeFlyCamera.enabled;
        // if(fromClick)
        //     toggleEl.isOn = freeFlyCamera.enabled;

        if(!freeFlyCamera.enabled){
            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Tab))
        {
            SetCamera();
            toggleEl.onValueChanged?.Invoke(true);
        }
    }
}
