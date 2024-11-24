import React, { useEffect, useRef, useState } from "react";
import logo from "./logo.png";

function App() {
  const videoRef = useRef(null);
  const [photo, setPhoto] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(true);

  const startCamera = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error("Error accessing camera:", err));
  };

  const takePicture = () => {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    if (video) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const photoData = canvas.toDataURL("image/png");
      setPhoto(photoData);

      // Call backend function to determine the object (placeholder here)
      fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: photoData }),
      })
        .then((response) => response.json())
        .then((data) => setResult(data.result || "Unknown object"))
        .catch((err) => console.error("Error:", err));
    }
  };

  useEffect(() => {
    // Simulate a 2-second loading screen
    const timer = setTimeout(() => {
      setLoading(false);
      startCamera();
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          backgroundColor: "#F7F4F2",
        }}
      >
        <img
          src={logo} // Replace with your image URL
          alt="Loading"
          style={{ marginBottom: "20px" }}
        />
        <span
          style={{
            color: "#37725F",
            fontSize: 96,
            marginLeft: 60,
            marginRight: 60,
          }}
        >
          {"“See”"}
        </span>
      </div>
    );
  }

  return (
    <div style={{ textAlign: "center" }}>
      <video
        ref={videoRef}
        autoPlay
        style={{ width: "100%", maxWidth: "500px" }}
        alt="Camera feed" // Alt text for the camera feed
      />
      <br />
      <button onClick={takePicture}>Take a Picture</button>
      {photo && (
        <div>
          <h3>Your Picture:</h3>
          <img
            src={photo}
            alt="Captured Photo"
            style={{ width: "100%", maxWidth: "500px" }}
          />
          <h3>Detected Object:</h3>
          <p>{result || "Processing..."}</p>
        </div>
      )}
    </div>
  );
}

export default App;
