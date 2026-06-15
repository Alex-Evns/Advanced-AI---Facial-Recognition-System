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

    preview.src = URL.createObjectURL(file);

    preview.style.display = "block";

    resultsDiv.innerHTML = "";

});

function getBarColour(probability) {

    const percentage = probability * 100;

    if (percentage >= 80) {

        return "#4CAF50";

    }

    if (percentage >= 60) {

        return "#FFC107";

    }

    return "#F44336";

}

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

        const predictions = data.predictions;

        const glassesColour = getBarColour(predictions.glasses);
        const hatColour = getBarColour(predictions.hat);
        const youngColour = getBarColour(predictions.young);

        resultsDiv.innerHTML = `

            <h2>Predictions</h2>

            <div class="prediction-card">

                <h3>Glasses</h3>

                <p>${(predictions.glasses * 100).toFixed(1)}%</p>

                <div class="bar">

                    <div
                        class="fill"
                        style="
                            width:${predictions.glasses * 100}%;
                            background:${glassesColour};
                        ">

                    </div>

                </div>

            </div>

            <div class="prediction-card">

                <h3>Hat</h3>

                <p>${(predictions.hat * 100).toFixed(1)}%</p>

                <div class="bar">

                    <div
                        class="fill"
                        style="
                            width:${predictions.hat * 100}%;
                            background:${hatColour};
                        ">

                    </div>

                </div>

            </div>

            <div class="prediction-card">

                <h3>Young</h3>

                <p>${(predictions.young * 100).toFixed(1)}%</p>

                <div class="bar">

                    <div
                        class="fill"
                        style="
                            width:${predictions.young * 100}%;
                            background:${youngColour};
                        ">

                    </div>

                </div>

            </div>

            <h2>Agent Analysis</h2>

                <h2>AI Analysis</h2>

    <div class="agent-card">

        <div class="agent-header">

            🤖 Facial Analysis Agent

        </div>

        <div class="agent-body">

            ${data.analysis.replace(/\n/g, "<br>")}

        </div>

    </div>

        `;

    }

    catch (error) {

        resultsDiv.innerHTML =
            "<p>Error analysing image.</p>";

    }

});