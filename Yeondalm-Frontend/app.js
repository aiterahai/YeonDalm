const dropzone = document.getElementById("dropzone");
const resultDiv = document.getElementById("result");
const imageDiv = document.getElementById("imagediv");

const refreshBtn = document.getElementById("refresh-btn");

dropzone.style.height = "439px"

function showGeneratingMessage() {
  const generatingMessage = document.createElement("p");
  generatingMessage.style.marginTop = "100px"
  generatingMessage.style.textAlign = "center"
  imageDiv.appendChild(generatingMessage);

  let dots = 1;
  const intervalId = setInterval(() => {
    generatingMessage.textContent = "인공지능 이미지 생성 중" + ".".repeat(dots);
    dots = dots % 3 + 1;
  }, 500);

  return intervalId;
}

    refreshBtn.addEventListener("click", () => {
      location.reload();
    });

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dropzone.style.border = "2px dashed #6B6B6B";
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
  dropzone.style.height = "50px";

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
      resultDiv.innerHTML = `1위 : ${items[0]} 2위 : ${items[1]} 3위 : ${items[2]}`
      const intervalId = showGeneratingMessage();

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
        // 이미지 로딩 완료 후 전송한 이미지와 받은 이미지 모두 표시
      img.onload = function () {
        clearInterval(intervalId);
        imageDiv.innerHTML = "";
        imageDiv.appendChild(img);

        // 전송한 이미지를 표시하는 img 요소 생성
        const sourceImg = new Image();
        sourceImg.src = URL.createObjectURL(files[0]);
        sourceImg.onload = function () {
          imageDiv.insertBefore(sourceImg, imageDiv.firstChild);
        };

        // 다운로드 버튼 생성
        const link = document.createElement("a");
        link.download = "result.png";
        link.id = "download-btn";
        link.href = imageUrl;
        link.textContent = "이미지 다운로드";
        resultDiv.insertAdjacentElement("afterend", link);

        const des = document.createElement("p");
        des.textContent = `입력된 사진을 ${firstItem}으로 FaceSwap한 사진입니다.`;
        des.style.textAlign = "center";
        link.insertAdjacentElement("afterend", des)
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

dropzone.addEventListener("click", () => {
  dropzone.style.height = "50px"
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = async (event) => {
    const files = event.target.files;
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
        resultDiv.innerHTML = `1위 : ${items[0]} 2위 : ${items[1]} 3위 : ${items[2]}`
        const intervalId = showGeneratingMessage();

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
          // 이미지 로딩 완료 후 전송한 이미지와 받은 이미지 모두 표시
      img.onload = function () {
        clearInterval(intervalId);
        imageDiv.innerHTML = "";
        imageDiv.appendChild(img);

        // 전송한 이미지를 표시하는 img 요소 생성
        const sourceImg = new Image();
        sourceImg.src = URL.createObjectURL(files[0]);
        sourceImg.onload = function () {
          imageDiv.insertBefore(sourceImg, imageDiv.firstChild);
        };

        // 다운로드 버튼 생성
        const link = document.createElement("a");
        link.download = "result.png";
        link.id = "download-btn";
        link.href = imageUrl;
        link.textContent = "이미지 다운로드";
        resultDiv.insertAdjacentElement("afterend", link);

        const des = document.createElement("p");
        des.textContent = "결과";
        link.insertAdjacentElement("afterend", des)
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
  };
  input.click();
});
