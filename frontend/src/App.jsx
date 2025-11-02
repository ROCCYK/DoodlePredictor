import React, { useRef, useState, useCallback, useEffect } from 'react'
import SignatureCanvas from 'react-signature-canvas'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Wand2, Eraser, Sparkles, Loader2 } from 'lucide-react'

function App() {
  const sigCanvas = useRef(null)
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const initialTimerRef = useRef(null)
  const intervalTimerRef = useRef(null)
  const isDrawingRef = useRef(false)
  const hasStartedRef = useRef(false)

  const clearCanvas = () => {
    sigCanvas.current.clear()
    setPredictions([])
    setError(null)
    // Clear any pending predictions
    if (initialTimerRef.current) {
      clearTimeout(initialTimerRef.current)
      initialTimerRef.current = null
    }
    if (intervalTimerRef.current) {
      clearInterval(intervalTimerRef.current)
      intervalTimerRef.current = null
    }
    hasStartedRef.current = false
    isDrawingRef.current = false
  }

  const makePrediction = useCallback(async () => {
    if (!sigCanvas.current || sigCanvas.current.isEmpty()) {
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Get the canvas as a data URL
      const canvas = sigCanvas.current.getCanvas()
      const imageData = canvas.toDataURL('image/png')

      // Send to backend
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData
        })
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }, [])

  // Handler when user starts drawing
  const handleDrawingStart = useCallback(() => {
    isDrawingRef.current = true

    // If this is the first stroke, start the prediction cycle
    if (!hasStartedRef.current) {
      hasStartedRef.current = true

      // Make first prediction after 1 second
      initialTimerRef.current = setTimeout(() => {
        makePrediction()

        // Then continue predicting every 1 second while drawing
        intervalTimerRef.current = setInterval(() => {
          if (isDrawingRef.current && !sigCanvas.current.isEmpty()) {
            makePrediction()
          }
        }, 1000)
      }, 1000)
    }
  }, [makePrediction])

  // Handler when user stops drawing (lifts mouse/touch)
  const handleDrawingEnd = useCallback(() => {
    isDrawingRef.current = false
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (initialTimerRef.current) {
        clearTimeout(initialTimerRef.current)
      }
      if (intervalTimerRef.current) {
        clearInterval(intervalTimerRef.current)
      }
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center gap-3">
            <Sparkles className="w-8 h-8" />
            <h1 className="text-4xl font-bold">Doodle Classifier</h1>
          </div>
          <p className="text-center mt-2 text-purple-100">
            Draw a doodle and let the AI predict what it is!
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Canvas Card */}
          <Card className="shadow-xl border-2 border-purple-100">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="w-5 h-5 text-purple-600" />
                Draw Here
              </CardTitle>
              <CardDescription>
                Use your mouse or touch to draw - predictions update automatically!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Canvas */}
              <div className="border-4 border-purple-300 rounded-lg overflow-hidden bg-white shadow-inner inline-block">
                <SignatureCanvas
                  ref={sigCanvas}
                  canvasProps={{
                    width: 500,
                    height: 500,
                    className: 'cursor-crosshair',
                    style: { width: '500px', height: '500px' }
                  }}
                  backgroundColor="#FFFFFF"
                  penColor="#000000"
                  minWidth={2}
                  maxWidth={4}
                  onBegin={handleDrawingStart}
                  onEnd={handleDrawingEnd}
                />
              </div>

              {/* Buttons */}
              <div className="flex gap-3">
                <Button
                  onClick={clearCanvas}
                  variant="outline"
                  size="lg"
                  className="flex-1 border-2"
                >
                  <Eraser className="mr-2 h-4 w-4" />
                  Clear Canvas
                </Button>
              </div>

              {/* Loading indicator for real-time predictions */}
              {loading && (
                <div className="flex items-center justify-center gap-2 p-3 bg-purple-50 border border-purple-200 rounded-lg">
                  <Loader2 className="h-4 w-4 animate-spin text-purple-600" />
                  <span className="text-sm text-purple-700">Analyzing your drawing...</span>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
                  <p className="font-medium">⚠️ {error}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Predictions Card */}
          <Card className="shadow-xl border-2 border-indigo-100">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-indigo-600" />
                Predictions
              </CardTitle>
              <CardDescription>
                Live predictions as you draw
              </CardDescription>
            </CardHeader>
            <CardContent>
              {predictions.length === 0 ? (
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <Wand2 className="w-16 h-16 mx-auto mb-4 text-purple-300" />
                    <p>Draw something to get predictions!</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {predictions.map((prediction, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border-2 border-purple-200 hover:border-purple-300 transition-all duration-200 hover:shadow-md"
                    >
                      <Badge
                        variant="default"
                        className="text-lg px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600"
                      >
                        #{index + 1}
                      </Badge>
                      <span className="text-xl font-semibold capitalize text-gray-800 flex-1">
                        {prediction.replace(/_/g, ' ')}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Section */}
        <Card className="mt-12 max-w-4xl mx-auto shadow-xl border-2 border-purple-100">
          <CardHeader>
            <CardTitle>About This Model</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground leading-relaxed">
              The architecture employed for this Convolutional Neural Network (CNN) doodle classifier
              is based on the <strong>MobileNetV1</strong> model. The classifier is trained using Google's
              <strong> Quick, Draw!</strong> dataset with <strong>340 different categories</strong> and
              over 50 million doodles. To enhance model performance and ensure robustness against variations
              in real-world doodling, the doodles are randomly augmented through rotations, shifts, shearing,
              zooming, and pixelation.
            </p>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6 mt-16">
        <div className="container mx-auto px-4 text-center text-sm">
          <p className="opacity-80">
            Dataset from Google's Quick, Draw! • Model: MobileNetV1 • Built with React, Vite, and Shadcn UI
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
