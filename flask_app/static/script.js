const img = document.getElementById("videoStream");

function updateStream() {
    img.src = "/video_feed?ts=" + new Date().getTime(); // avoid caching
}

img.onerror = () => {
    console.warn("Stream error, retrying...");
    setTimeout(updateStream, 1000); // retry after 1s
};

setInterval(updateStream, 100); // update every 100ms (~10 FPS)
