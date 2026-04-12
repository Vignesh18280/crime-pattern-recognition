// src/lib/api.ts

const API_BASE_URL = "http://localhost:8080/api";

export async function uploadFile(file: File): Promise<{ filename: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "File upload failed");
  }

  return response.json();
}

export async function predict(file1: string, file2: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      uploadedFile1: file1,
      uploadedFile2: file2,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "Prediction failed");
  }

  return response.json();
}
