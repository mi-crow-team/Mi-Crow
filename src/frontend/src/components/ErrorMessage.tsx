"use client";

import { getErrorMessage, isApiError } from "@/lib/errors";

type ErrorMessageProps = {
  error: unknown;
  className?: string;
};

export function ErrorMessage({ error, className = "" }: ErrorMessageProps) {
  const message = getErrorMessage(error);
  const isApi = isApiError(error);

  return (
    <div
      className={`rounded-md border p-3 text-sm ${
        isApi && error.statusCode === 404
          ? "border-yellow-200 bg-yellow-50 text-yellow-800"
          : isApi && error.statusCode && error.statusCode >= 500
            ? "border-red-200 bg-red-50 text-red-800"
            : "border-red-200 bg-red-50 text-red-800"
      } ${className}`}
    >
      <div className="font-medium">Error</div>
      <div className="mt-1">{message}</div>
    </div>
  );
}

