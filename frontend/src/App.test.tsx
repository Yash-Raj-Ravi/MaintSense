import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

function response(body: unknown, status = 200, contentType = 'application/json'): Response {
  const bodyText = typeof body === 'string' ? body : JSON.stringify(body)
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': contentType }),
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(bodyText),
  } as unknown as Response
}

describe('MaintSense dashboard', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_API_BASE_URL', '/api')
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const path = String(input)
      if (path.endsWith('/health')) {
        return Promise.resolve(response({ status: 'ok', model_available: true }))
      }
      if (path.endsWith('/model-info')) {
        return Promise.resolve(response({
          model_type: 'RandomForestClassifier',
          threshold: 0.4,
          required_features: [
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
          ],
          scope: 'Machine-failure classification on the synthetic AI4I 2020 dataset.',
          model_available: true,
        }))
      }
      return Promise.resolve(response({ failure_probability: 0.52, predicted_class: 1, threshold: 0.4 }))
    }))
  })

  it('shows the dashboard and sends machine records to the prediction API', async () => {
    render(<App />)

    expect(await screen.findByText('API connected')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Machine Assessment' })).toBeInTheDocument()
    expect(screen.getByText('FINAL TEST')).toBeInTheDocument()
    expect(screen.getByText(/has not been validated as a real-world early-warning system/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /run assessment/i }))

    await screen.findByText('Failure class')
    expect(fetch).toHaveBeenCalledWith('/api/predict', expect.objectContaining({ method: 'POST' }))
    const [, options] = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith('/predict'))!
    expect(JSON.parse(String(options?.body))).toEqual({
      Type: 'M',
      'Air temperature [K]': 300,
      'Process temperature [K]': 310,
      'Rotational speed [rpm]': 1500,
      'Torque [Nm]': 40,
      'Tool wear [min]': 120,
    })
    expect(screen.getByText('52.0')).toBeInTheDocument()
    expect(screen.queryByText('LOCAL ENVIRONMENT')).not.toBeInTheDocument()
  })

  it('keeps the API connected when health fails but model information succeeds', async () => {
    vi.mocked(fetch)
      .mockRejectedValueOnce(new TypeError('ERR_BLOCKED_BY_CLIENT'))
      .mockResolvedValueOnce(response({
        model_type: 'RandomForestClassifier',
        threshold: 0.4,
        required_features: ['Type'],
        scope: 'Synthetic data',
        model_available: true,
      }))

    render(<App />)

    expect(await screen.findByText('API connected')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Connected' })).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(
      'Health check unavailable: Unable to reach the prediction service.',
    )
    expect(screen.getByText('RandomForestClassifier')).toBeInTheDocument()
  })

  it('clears a partial-status error after a successful status refresh', async () => {
    vi.mocked(fetch)
      .mockRejectedValueOnce(new TypeError('ERR_BLOCKED_BY_CLIENT'))
      .mockResolvedValueOnce(response({
        model_type: 'RandomForestClassifier',
        threshold: 0.4,
        required_features: ['Type'],
        scope: 'Synthetic data',
        model_available: true,
      }))
      .mockResolvedValueOnce(response({ status: 'ok', model_available: true }))
      .mockResolvedValueOnce(response({
        model_type: 'RandomForestClassifier',
        threshold: 0.4,
        required_features: ['Type'],
        scope: 'Synthetic data',
        model_available: true,
      }))

    render(<App />)
    expect(await screen.findByRole('status')).toHaveTextContent('Health check unavailable')

    fireEvent.click(screen.getByRole('button', { name: 'Refresh API status' }))

    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
    expect(screen.getByText('API connected')).toBeInTheDocument()
  })

  it('shows validation feedback for empty values without making an API call', async () => {
    render(<App />)
    await screen.findByText('API connected')
    fireEvent.change(screen.getByLabelText('Air temperature'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /run assessment/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Air temperature must be a finite number.')
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2))
  })

  it('shows a friendly connection message when the Vite proxy cannot reach FastAPI', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(response({ status: 'ok', model_available: true }))
      .mockResolvedValueOnce(response({
        model_type: 'RandomForestClassifier',
        threshold: 0.4,
        required_features: ['Type'],
        scope: 'Synthetic data',
        model_available: true,
      }))
      .mockResolvedValueOnce(response('', 500, 'text/plain'))

    render(<App />)
    await screen.findByText('API connected')
    fireEvent.click(screen.getByRole('button', { name: /run assessment/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Unable to reach the prediction service. Check that the backend is running and try again.',
    )
  })
})
