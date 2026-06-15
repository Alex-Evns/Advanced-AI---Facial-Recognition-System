const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

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