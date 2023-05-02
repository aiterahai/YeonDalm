const dropzone = document.getElementById("dropzone");
const resultDiv = document.getElementById("result");
const imageDiv = document.getElementById("imagediv");

const refreshBtn = document.getElementById("refresh-btn");

    refreshBtn.addEventListener("click", () => {
      location.reload();
    });

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropzone.style.border = "2px dashed #aaa";
});

dropzone.addEventListener("dragleave", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropzone.style.border = "2px dashed #ccc";
});

dropzone.addEventListener("drop", async (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropzone.style.border = "2px dashed #ccc";

  const files = event.dataTransfer.files;
  if (dropzone.hasChildNodes()) {
    dropzone.removeChild(dropzone.firstChild);
  }

  const formData = new FormData();
  formData.append("image", files[0], files[0].name);

  try {
    const response = await fetch("http://localhost:8000/pred", {
      method: "POST",
      body: formData,
    });
    if (response.ok) {
      const items = await response.json();
      const firstItem = items[0];

      const sourceFormData = new FormData();
      sourceFormData.append("target", files[0], files[0].name);
      resultDiv.innerHTML = items.join(", ");

      const sourceResponse = await fetch(
        `http://localhost:8000/image/${firstItem}`,
        {
          method: "POST",
          body: sourceFormData,
        }
      );

      if (sourceResponse.ok) {
        const imageUrl = URL.createObjectURL(await sourceResponse.blob());
        const img = new Image();
        img.src = imageUrl;

        // 이미 다운로드 버튼이 생성되었다면 제거
        const downloadBtn = document.getElementById("download-btn");
        if (downloadBtn) {
          downloadBtn.remove();
        }

        // 이미지 로딩 완료 후 다운로드 버튼 생성
        img.onload = function() {
          imageDiv.innerHTML = "";
          imageDiv.appendChild(img);
          const link = document.createElement('a');
          link.download = 'result.png';
          link.id = "download-btn";
          link.href = imageUrl;
          link.textContent = '다운로드';
          resultDiv.insertAdjacentElement('afterend', link);
        };
      } else {
        console.log("Failed to upload image");
      }
    } else {
      console.log("Prediction failed");
    }
  } catch (error) {
    console.log(error);
  }
});
