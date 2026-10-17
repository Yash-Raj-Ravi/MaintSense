import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
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
  })

  it('shows validation feedback for empty values without making an API call', async () => {
    render(<App />)
    await screen.findByText('API connected')
    fireEvent.change(screen.getByLabelText('Air temperature'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /run assessment/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Air temperature must be a finite number.')
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2))
  })
})
