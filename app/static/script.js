const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyseButton = document.getElementById("analyseButton");
const resultsDiv = document.getElementById("results");

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (!file) {
        preview.style.display = "none";
        return;
    }

    const imageUrl = URL.createObjectURL(file);

    preview.src = imageUrl;
    preview.style.display = "block";
});

analyseButton.addEventListener("click", async function () {
    const file = imageInput.files[0];

    if (!file) {
        resultsDiv.innerHTML = "<p>Please choose an image first.</p>";
        return;
    }

    resultsDiv.innerHTML = "<p>Analysing image...</p>";

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            resultsDiv.innerHTML = `<p>${data.error}</p>`;
            return;
        }

        const predictions = data.predictions;

        resultsDiv.innerHTML = `
        <h2>Predictions</h2>

        <div class="prediction-card">
            <p>Glasses: ${(predictions.glasses * 100).toFixed(1)}%</p>
            <div class="bar">
                <div class="fill" style="width: ${predictions.glasses * 100}%"></div>
            </div>
        </div>

        <div class="prediction-card">
            <p>Hat: ${(predictions.hat * 100).toFixed(1)}%</p>
            <div class="bar">
                <div class="fill" style="width: ${predictions.hat * 100}%"></div>
            </div>
        </div>

        <div class="prediction-card">
            <p>Young: ${(predictions.young * 100).toFixed(1)}%</p>
            <div class="bar">
                <div class="fill" style="width: ${predictions.young * 100}%"></div>
            </div>
        </div>

        <h2>Agent Analysis</h2>

        <pre>${data.analysis}</pre>
    `;

    } catch (error) {
        resultsDiv.innerHTML = "<p>An error occurred while analysing the image.</p>";
        console.error(error);
    }
});