using System.Collections.Generic;
using Dreamteck.Splines;
using Unity.MLAgents.Areas;
using UnityEngine;

public class TrackGenerator : MonoBehaviour
{
    [System.Serializable]
    public struct TrackPiece
    {
        public GameObject prefab;
        public float angle;

        [HideInInspector]
        public GameObject go;

        public TrackPiece(GameObject _prefab, float _angle)
        {
            prefab = _prefab;
            angle = _angle;
            go = null;
        }
    }

    public TrackPiece[] trackPieces;
    public float trackPieceSize = 2;
    public int activeTracksAtOnce = 4;

    public List<TrackPiece> track = new List<TrackPiece>();

    public int triesPerTrack = 2;
    private int currentTry = 1;

    public GameObject checkpointParent;
    public GameObject checkpointMarker;
    public List<SplineComputer> checkpoints = new List<SplineComputer>();

    public TrainingReplicator trainingReplicator;

    [System.Serializable]
    public struct MaterialsInspector
    {
        public TrainingReplicator.RoadColor roadColor;
        public Material material;
    };

    public MaterialsInspector[] floorMaterialsInspector;
    public MaterialsInspector[] roadMaterialsInspector;
    public Dictionary<TrainingReplicator.RoadColor, Material> floorMaterials =
        new Dictionary<TrainingReplicator.RoadColor, Material>();
    public Dictionary<TrainingReplicator.RoadColor, Material> roadMaterials =
        new Dictionary<TrainingReplicator.RoadColor, Material>();

    public void Start()
    {
        foreach (MaterialsInspector fmi in floorMaterialsInspector)
        {
            floorMaterials[fmi.roadColor] = fmi.material;
        }
        foreach (MaterialsInspector rmi in roadMaterialsInspector)
        {
            roadMaterials[rmi.roadColor] = rmi.material;
        }

        foreach (TrackPiece tp in trackPieces)
        {
            GameObject floor = tp.prefab.transform.Find("Floor").gameObject;
            GameObject road = tp.prefab.transform.Find("Track").gameObject;

            MeshRenderer floorMesh = floor.GetComponent<MeshRenderer>();
            floorMesh.material = floorMaterials[trainingReplicator.roadColor];

            MeshRenderer roadMesh = road.GetComponent<MeshRenderer>();
            roadMesh.material = roadMaterials[trainingReplicator.roadColor];

            PathGenerator roadPath = road.GetComponent<PathGenerator>();
            roadPath.size = trainingReplicator.roadSize;
        }
    }

    public void ResetTrack()
    {
        if (track.Count != 0 && currentTry < triesPerTrack)
        {
            foreach (TrackPiece trackPiece in track)
            {
                trackPiece.go.SetActive(false);
            }

            for (int i = 0; i < activeTracksAtOnce; i++)
            {
                PlacePiece(i);
            }

            currentTry++;
            return;
        }

        RemoveCheckpoints();
        foreach (TrackPiece trackPiece in track)
        {
            Destroy(trackPiece.go);
        }

        track.Clear();
        foreach (TrackPiece piece in trackPieces)
        {
            if (piece.angle == 0)
            {
                track.Add(piece);
                break;
            }
        }

        PlacePiece(0);

        for (int i = 0; i < activeTracksAtOnce - 1; i++)
            GenerateTrackPiece();

        currentTry = 1;
    }

    void RemoveCheckpoints()
    {
        checkpoints.Clear();
    }

    void GenerateTrackPiece()
    {
        int lastIndex = track.Count - 1;
        if (track.Count >= 2 && track[lastIndex].angle == 0 && track[lastIndex - 1].angle == 0)
            track.Add(trackPieces[Random.Range(0, trackPieces.Length - 1) + 1]);
        else
            track.Add(trackPieces[Random.Range(0, trackPieces.Length)]);

        PlacePiece(track.Count - 1);
    }

    void PlacePiece(int index)
    {
        TrackPiece currentTrackPiece = track[index];

        if (currentTrackPiece.go != null)
        {
            track[index].go.SetActive(true);
            return;
        }

        float angle = 0;
        Vector3 pos = transform.parent.transform.localPosition;
        if (index != 0)
        {
            TrackPiece previousTrackPiece = track[index - 1];
            Track previousTrack = previousTrackPiece.go.GetComponent<Track>();
            angle = previousTrack.continueAngle;

            float rad = angle * Mathf.Deg2Rad;
            Vector3 dir = new Vector3(Mathf.Sin(rad), 0, Mathf.Cos(rad)) * trackPieceSize;

            pos = previousTrackPiece.go.transform.position + dir;
        }

        Quaternion rotation = Quaternion.Euler(0, angle, 0);
        GameObject go = Instantiate(currentTrackPiece.prefab, pos, rotation, this.transform);
        currentTrackPiece.go = go;
        track[index] = currentTrackPiece;

        Track currentTrack = go.GetComponent<Track>();
        currentTrack.continueAngle = angle + track[index].angle;

        AddCheckpoint(currentTrackPiece);
    }

    public void UpdateTrack(int index)
    {
        for (int i = 0; i < index - 1; i++)
        {
            track[i].go.SetActive(false);
        }

        int newPieceIndex = index + activeTracksAtOnce - 1;
        if (newPieceIndex < track.Count)
            PlacePiece(newPieceIndex);
        else
            GenerateTrackPiece();
    }

    void AddCheckpoint(TrackPiece track)
    {
        SplineComputer checkpoint = track
            .go.transform.Find("Line")
            .gameObject.GetComponent<SplineComputer>();
        checkpoints.Add(checkpoint);
    }
}
