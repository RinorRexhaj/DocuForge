export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};

export const formatTimestamp = (timestamp: string | Date): string => {
  const date = new Date(timestamp);
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
};

export const validateImageFile = (
  file: File
): { valid: boolean; error?: string } => {
  const MAX_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

  if (!ALLOWED_TYPES.includes(file.type)) {
    return {
      valid: false,
      error: "Invalid file type. Only JPG and PNG images are allowed.",
    };
  }

  if (file.size > MAX_SIZE) {
    return {
      valid: false,
      error: `File size exceeds 10MB limit. Current size: ${formatFileSize(
        file.size
      )}`,
    };
  }

  return { valid: true };
};

export const downloadBase64Image = (
  base64Data: string,
  filename: string
): void => {
  const link = document.createElement("a");
  link.href = base64Data;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const getConfidenceLevel = (
  probability: number
): {
  level: "low" | "medium" | "high";
  color: string;
  description: string;
} => {
  if (probability < 0.3) {
    return {
      level: "low",
      color: "text-green-600",
      description: "Low probability of tampering",
    };
  } else if (probability < 0.7) {
    return {
      level: "medium",
      color: "text-yellow-600",
      description: "Moderate probability of tampering",
    };
  } else {
    return {
      level: "high",
      color: "text-red-600",
      description: "High probability of tampering",
    };
  }
};
