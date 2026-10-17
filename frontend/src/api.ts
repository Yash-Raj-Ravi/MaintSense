export type MachineType = 'H' | 'L' | 'M'

export interface HealthResponse {
  status: string
  model_available: boolean
}

export interface ModelInfoResponse {
  model_type: string
  threshold: number
  required_features: string[]
  scope: string
  model_available: boolean
}

export interface PredictionRequest {
  Type: MachineType
  'Air temperature [K]': number
  'Process temperature [K]': number
  'Rotational speed [rpm]': number
  'Torque [Nm]': number
  'Tool wear [min]': number
}

export interface PredictionResponse {
  failure_probability: number
  predicted_class: 0 | 1
  threshold: number
}

export class ApiError extends Error {
  readonly status: number

  constructor(
    message: string,
    status: number,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

const CONNECTION_ERROR_MESSAGE =
  'Unable to reach the prediction service. Check that the backend is running and try again.'

export function apiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch {
    throw new NetworkError(CONNECTION_ERROR_MESSAGE)
  }

  if (!response.ok) {
    let message = `API request failed (${response.status}).`
    let bodyText = ''
    try {
      bodyText = await response.text()
    } catch {
      // Keep the status-based message if the response body cannot be read.
    }

    // Vite's dev proxy emits an empty text/plain 500 when its upstream is
    // unreachable. A normal backend 500 has a response body, so it remains an
    // ApiError instead of being mislabeled as a connectivity failure.
    const contentType = response.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase()
    if (response.status === 500 && contentType === 'text/plain' && bodyText.trim() === '') {
      throw new NetworkError(CONNECTION_ERROR_MESSAGE)
    }

    try {
      const body: unknown = JSON.parse(bodyText)
      if (typeof body === 'object' && body !== null && 'detail' in body) {
        const detail = (body as { detail: unknown }).detail
        if (typeof detail === 'string') message = detail
        else if (Array.isArray(detail)) {
          message = detail
            .map((item) => {
              if (typeof item === 'object' && item !== null && 'msg' in item) {
                return String((item as { msg: unknown }).msg)
              }
              return 'The request contains invalid values.'
            })
            .join(' ')
        }
      }
    } catch {
      // Keep the status-based message if the response is not JSON.
    }
    throw new ApiError(message, response.status)
  }

  return (await response.json()) as T
}

export const getHealth = () => request<HealthResponse>('/health')
export const getModelInfo = () => request<ModelInfoResponse>('/model-info')

export const predict = (payload: PredictionRequest) =>
  request<PredictionResponse>('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
