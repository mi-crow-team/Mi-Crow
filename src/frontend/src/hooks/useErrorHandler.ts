import { useState, useCallback } from "react";
import { handleApiError, getErrorMessage } from "@/lib/errors";

/**
 * Hook for handling errors in components.
 */
export function useErrorHandler() {
  const [error, setError] = useState<unknown>(null);

  const handleError = useCallback((err: unknown) => {
    const apiError = handleApiError(err);
    setError(apiError);
    console.error("Error handled:", apiError);
    return apiError;
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const errorMessage = error ? getErrorMessage(error) : null;

  return {
    error,
    errorMessage,
    handleError,
    clearError,
  };
}

