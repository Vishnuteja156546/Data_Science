function initUpload() {
    const fileInput = $("fileInput");
    const dropZone = $("dropZone");
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) handleUpload(fileInput.files[0]);
    });
    ["dragover", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, event => {
            event.preventDefault();
            if (eventName === "drop" && event.dataTransfer.files.length) {
                handleUpload(event.dataTransfer.files[0]);
            }
        });
    });
}

async function handleUpload(file) {
    setStatus("uploadStatus", "Uploading and profiling dataset...");
    try {
        const data = await API.upload(file);
        API.sessionId = data.session_id;
        renderDashboard(data);
        setStatus("uploadStatus", `Loaded ${data.filename}`, "ok");
    } catch (error) {
        setStatus("uploadStatus", error.message, "error");
    }
}
