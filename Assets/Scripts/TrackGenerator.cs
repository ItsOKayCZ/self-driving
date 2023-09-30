using System.Collections.Generic;
using UnityEngine;

public class TrackGenerator : MonoBehaviour
{
	[System.Serializable]
	public struct TrackPiece
	{
		public GameObject prefab;
		public float angle;
		[HideInInspector] public GameObject go;

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

	private List<TrackPiece> track = new List<TrackPiece>();

	public int triesPerTrack = 2;
	private int currentTry = 1;

	public GameObject checkpointParent;
	public GameObject checkpointMarker;
	public List<GameObject> checkpoints = new List<GameObject>();

    public void ResetTrack()
	{
		if(track.Count != 0 && currentTry < triesPerTrack)
		{
			foreach(TrackPiece trackPiece in track)
			{
				trackPiece.go.SetActive(false);
			}

			for(int i = 0; i < activeTracksAtOnce; i++)
			{
				PlacePiece(i);
			}

            Debug.Log("Resetting track");

			currentTry++;
			return;
		}

		Debug.Log("Generating new track");

		RemoveCheckpoints();
		foreach(TrackPiece trackPiece in track)
		{
			Destroy(trackPiece.go);
		}

		track.Clear();
		foreach(TrackPiece piece in trackPieces)
		{
			if(piece.angle == 0)
			{
				track.Add(piece);
				break;
			}
		}

		PlacePiece(0);

		for (int i = 0; i < activeTracksAtOnce - 1; i++) GenerateTrackPiece();

		currentTry = 1;
	}

	void RemoveCheckpoints()
	{
		foreach (GameObject checkpoint in checkpoints)
			Destroy(checkpoint);

		checkpoints.Clear();
	}

	void GenerateTrackPiece()
	{
		int lastIndex = track.Count - 1;
		if(track.Count >= 2 && track[lastIndex].angle == 0 && track[lastIndex - 1].angle == 0)
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
		if(index != 0)
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

		{
			PlaceCheckpoint(pos);
			//if(index == 0)
			//{
			//	PlaceCheckpoint(pos);
			//} else
			//{
   //             float newAngle = angle + currentTrackPiece.angle;
   //             float rad = newAngle * Mathf.Deg2Rad;

   //             Vector3 dir = new Vector3(Mathf.Sin(rad), 0, Mathf.Cos(rad)) * trackPieceSize / 2f;

   //             PlaceCheckpoint(pos, dir);
			//}
		}

		Track currentTrack = go.GetComponent<Track>();
		currentTrack.continueAngle = angle + track[index].angle;
	}

	public void UpdateTrack(int index)
	{
		for(int i = 0; i < index - 1; i++)
		{
			track[i].go.SetActive(false);
		}

		int newPieceIndex = index + activeTracksAtOnce - 1;
		if (newPieceIndex < track.Count)
			PlacePiece(newPieceIndex);
		else
            GenerateTrackPiece();
	}

	void PlaceCheckpoint(Vector3 position, Vector3 offset = new Vector3())
	{

		Vector3 newPos = position + Vector3.up * 15 + offset;
	

        GameObject checkpoint = Instantiate(checkpointMarker, newPos, Quaternion.identity, checkpointParent.transform);

        checkpoints.Add(checkpoint);
	}
}
