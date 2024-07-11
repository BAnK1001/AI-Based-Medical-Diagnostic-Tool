document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("upload-form");
  const fileUpload = document.getElementById("file-upload");
  const uploadArea = document.getElementById("upload-area");
  const fileName = document.getElementById("file-name");
  const submitButton = document.querySelector("button[type='submit']");
  const imagePreview = document.createElement("img");
  imagePreview.className = "img-fluid mt-3 rounded";

  function updatePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      uploadArea.innerHTML = ""; // Clear existing content
      uploadArea.appendChild(imagePreview);
    };
    reader.readAsDataURL(file);
  }

  function handleFiles(files) {
    if (files.length > 0) {
      fileName.textContent = files[0].name;
      submitButton.disabled = false;
      updatePreview(files[0]);
    }
  }

  uploadArea.addEventListener("click", () => fileUpload.click());

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("bg-light");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("bg-light");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("bg-light");
    handleFiles(e.dataTransfer.files);
  });

  fileUpload.addEventListener("change", (e) => {
    handleFiles(e.target.files);
  });

  uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitButton.disabled = true;
    submitButton.innerHTML =
      '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';

    const formData = new FormData(uploadForm);

    fetch("/", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.text())
      .then((html) => {
        document.body.innerHTML = html;
        const scripts = document.body.getElementsByTagName("script");
        for (let script of scripts) {
          eval(script.innerHTML);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        submitButton.disabled = false;
        submitButton.textContent = "Upload and Classify";
      });
  });
});
