import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getHealth, getModelInfo, NetworkError, predict } from './api'

function response(
  body: unknown,
  status = 200,
  contentType = 'application/json',
): Response {
  const bodyText = typeof body === 'string' ? body : JSON.stringify(body)
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': contentType }),
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(bodyText),
  } as unknown as Response
}

describe('MaintSense API client', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_API_BASE_URL', '/api/v2/')
    vi.stubGlobal('fetch', vi.fn())
  })

  it('uses the documented health and model-info routes', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(response({ status: 'ok', model_available: true }))
      .mockResolvedValueOnce(response({
        model_type: 'RandomForestClassifier',
        threshold: 0.4,
        required_features: ['Type'],
        scope: 'Synthetic data',
        model_available: true,
      }))

    await getHealth()
    await getModelInfo()

    expect(fetch).toHaveBeenNthCalledWith(1, '/api/v2/health', expect.any(Object))
    expect(fetch).toHaveBeenNthCalledWith(2, '/api/v2/model-info', expect.any(Object))
  })

  it('posts the exact backend request feature names and returns the response', async () => {
    const result = { failure_probability: 0.25, predicted_class: 0 as const, threshold: 0.4 }
    vi.mocked(fetch).mockResolvedValueOnce(response(result))
    const payload = {
      Type: 'M' as const,
      'Air temperature [K]': 300,
      'Process temperature [K]': 310,
      'Rotational speed [rpm]': 1500,
      'Torque [Nm]': 40,
      'Tool wear [min]': 120,
    }

    await expect(predict(payload)).resolves.toEqual(result)

    expect(fetch).toHaveBeenCalledWith('/api/v2/predict', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify(payload),
    }))
  })

  it('preserves FastAPI validation errors', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response({
      detail: [{ msg: 'Field required' }],
    }, 422))

    await expect(getHealth()).rejects.toMatchObject({
      name: 'ApiError',
      status: 422,
      message: 'Field required',
    })
  })

  it('preserves model-unavailable API errors', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response({ detail: 'model unavailable' }, 503))

    await expect(getHealth()).rejects.toMatchObject({
      name: 'ApiError',
      status: 503,
      message: 'model unavailable',
    })
  })

  it('classifies Vite empty text proxy failures as a friendly network error', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response('', 500, 'text/plain'))

    await expect(getHealth()).rejects.toMatchObject({
      name: 'NetworkError',
      message: 'Unable to reach the prediction service. Check that the backend is running and try again.',
    })
  })

  it('keeps a genuine backend 500 as an HTTP API error', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response('Internal Server Error', 500, 'text/plain'))

    await expect(getHealth()).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      message: 'API request failed (500).',
    })
  })

  it('classifies fetch failures as network errors', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError('fetch failed'))

    await expect(getHealth()).rejects.toBeInstanceOf(NetworkError)
  })
})
