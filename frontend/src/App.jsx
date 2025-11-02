import React, { useRef, useState, useCallback, useEffect } from 'react'
import SignatureCanvas from 'react-signature-canvas'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Wand2, Eraser, Sparkles, Loader2, Play, Trophy, Timer, Target, SkipForward, CheckCircle2, XCircle } from 'lucide-react'

// Game states
const GAME_STATES = {
  IDLE: 'idle',
  PLAYING: 'playing',
  WON: 'won',
  LOST: 'lost'
}

function App() {
  const sigCanvas = useRef(null)

  // Game state
  const [gameState, setGameState] = useState(GAME_STATES.IDLE)
  const [targetObject, setTargetObject] = useState(null)
  const [score, setScore] = useState(0)
  const [roundsPlayed, setRoundsPlayed] = useState(0)
  const [timeLeft, setTimeLeft] = useState(20)
  const [categories, setCategories] = useState([])
  const [timeTaken, setTimeTaken] = useState(0)

  // Prediction state
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Refs for timers and drawing
  const initialTimerRef = useRef(null)
  const intervalTimerRef = useRef(null)
  const countdownTimerRef = useRef(null)
  const isDrawingRef = useRef(false)
  const hasStartedDrawingRef = useRef(false)
  const roundStartTimeRef = useRef(null)

  // Fetch categories on mount
  useEffect(() => {
    fetchCategories()
  }, [])

  const fetchCategories = async () => {
    try {
      const response = await fetch('http://localhost:8000/categories')
      const data = await response.json()
      setCategories(data.categories)
    } catch (err) {
      console.error('Failed to fetch categories:', err)
    }
  }

  const getRandomCategory = () => {
    if (categories.length === 0) return null
    const randomIndex = Math.floor(Math.random() * categories.length)
    return categories[randomIndex]
  }

  const startNewRound = () => {
    const newTarget = getRandomCategory()
    if (!newTarget) return

    // Reset everything for new round
    if (sigCanvas.current) {
      sigCanvas.current.clear()
    }
    setPredictions([])
    setError(null)
    setTargetObject(newTarget)
    setTimeLeft(20)
    setGameState(GAME_STATES.PLAYING)
    hasStartedDrawingRef.current = false
    roundStartTimeRef.current = Date.now()

    // Clear any existing timers
    if (initialTimerRef.current) clearTimeout(initialTimerRef.current)
    if (intervalTimerRef.current) clearInterval(intervalTimerRef.current)
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current)

    // Start countdown
    countdownTimerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          handleTimeUp()
          return 0
        }
        return prev - 1
      })
    }, 1000)
  }

  const startGame = () => {
    setScore(0)
    setRoundsPlayed(0)
    startNewRound()
  }

  const handleTimeUp = () => {
    setGameState(GAME_STATES.LOST)
    clearAllTimers()
  }

  const handleWin = () => {
    const endTime = Date.now()
    const timeTakenSeconds = Math.round((endTime - roundStartTimeRef.current) / 1000)
    setTimeTaken(timeTakenSeconds)
    setGameState(GAME_STATES.WON)
    setScore(prev => prev + 1)
    clearAllTimers()
  }

  const clearAllTimers = () => {
    if (initialTimerRef.current) {
      clearTimeout(initialTimerRef.current)
      initialTimerRef.current = null
    }
    if (intervalTimerRef.current) {
      clearInterval(intervalTimerRef.current)
      intervalTimerRef.current = null
    }
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current)
      countdownTimerRef.current = null
    }
    hasStartedDrawingRef.current = false
    isDrawingRef.current = false
  }

  const nextRound = () => {
    setRoundsPlayed(prev => prev + 1)
    startNewRound()
  }

  const skipRound = () => {
    setGameState(GAME_STATES.LOST)
    setRoundsPlayed(prev => prev + 1)
    clearAllTimers()
  }

  const makePrediction = useCallback(async () => {
    if (!sigCanvas.current || sigCanvas.current.isEmpty() || gameState !== GAME_STATES.PLAYING) {
      return
    }

    setLoading(true)
    setError(null)

    try {
      const canvas = sigCanvas.current.getCanvas()
      const imageData = canvas.toDataURL('image/png')

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

      // Check if target is in predictions (case insensitive, handle underscores)
      const normalizedTarget = targetObject.toLowerCase().replace(/ /g, '_')
      const normalizedPredictions = data.predictions.map(p => p.toLowerCase())

      if (normalizedPredictions.includes(normalizedTarget)) {
        handleWin()
      }
    } catch (err) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }, [gameState, targetObject])

  const handleDrawingStart = useCallback(() => {
    if (gameState !== GAME_STATES.PLAYING) return

    isDrawingRef.current = true

    if (!hasStartedDrawingRef.current) {
      hasStartedDrawingRef.current = true

      // Make first prediction after 1 second
      initialTimerRef.current = setTimeout(() => {
        makePrediction()

        // Then continue predicting every 1 second
        intervalTimerRef.current = setInterval(() => {
          if (!sigCanvas.current?.isEmpty()) {
            makePrediction()
          }
        }, 1000)
      }, 1000)
    }
  }, [gameState, makePrediction])

  const handleDrawingEnd = useCallback(() => {
    isDrawingRef.current = false
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearAllTimers()
    }
  }, [])

  // Auto-advance after winning
  useEffect(() => {
    if (gameState === GAME_STATES.WON) {
      const timer = setTimeout(() => {
        nextRound()
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [gameState])

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Sparkles className="w-8 h-8" />
              <h1 className="text-3xl font-bold">Quick, Draw!</h1>
            </div>
            {gameState !== GAME_STATES.IDLE && (
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <Trophy className="w-5 h-5" />
                  <span className="text-xl font-bold">{score}/{roundsPlayed}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {gameState === GAME_STATES.IDLE ? (
          // Start Screen
          <Card className="max-w-2xl mx-auto shadow-2xl border-2 border-purple-200">
            <CardContent className="p-12 text-center space-y-6">
              <Sparkles className="w-20 h-20 mx-auto text-purple-600" />
              <h2 className="text-4xl font-bold text-gray-800">Quick, Draw!</h2>
              <p className="text-xl text-gray-600">
                Can you draw fast enough for the AI to recognize your drawings?
              </p>
              <div className="space-y-2 text-left bg-purple-50 p-6 rounded-lg">
                <h3 className="font-bold text-lg mb-3">How to Play:</h3>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">1.</span>
                  You'll be shown what to draw
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">2.</span>
                  Draw it as fast as you can!
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">3.</span>
                  The AI will try to guess what you're drawing
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">4.</span>
                  You have 20 seconds per drawing
                </p>
              </div>
              <Button
                onClick={startGame}
                size="lg"
                className="text-xl px-12 py-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
              >
                <Play className="mr-2 h-6 w-6" />
                Let's Draw!
              </Button>
            </CardContent>
          </Card>
        ) : (
          // Game Screen
          <div className="space-y-6">
            {/* Target and Timer */}
            <Card className="max-w-4xl mx-auto shadow-xl border-2 border-purple-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <Target className="w-8 h-8 text-purple-600" />
                    <div>
                      <p className="text-sm text-gray-500 uppercase tracking-wide">Draw this:</p>
                      <p className="text-4xl font-bold text-gray-800 capitalize">
                        {targetObject?.replace(/_/g, ' ')}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Timer className={`w-8 h-8 ${timeLeft <= 5 ? 'text-red-500 animate-pulse' : 'text-purple-600'}`} />
                    <span className={`text-5xl font-bold ${timeLeft <= 5 ? 'text-red-500' : 'text-gray-800'}`}>
                      {timeLeft}s
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
              {/* Canvas Card */}
              <Card className="shadow-xl border-2 border-purple-100">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Wand2 className="w-5 h-5 text-purple-600" />
                    Draw Here
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Canvas */}
                  <div className="border-4 border-purple-300 rounded-lg overflow-hidden bg-white shadow-inner">
                    <SignatureCanvas
                      ref={sigCanvas}
                      canvasProps={{
                        width: 500,
                        height: 500,
                        className: 'cursor-crosshair',
                        style: { width: '100%', height: 'auto', maxWidth: '500px' }
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
                      onClick={() => {
                        if (sigCanvas.current) sigCanvas.current.clear()
                        setPredictions([])
                      }}
                      variant="outline"
                      size="lg"
                      className="flex-1 border-2"
                      disabled={gameState !== GAME_STATES.PLAYING}
                    >
                      <Eraser className="mr-2 h-4 w-4" />
                      Clear
                    </Button>
                    <Button
                      onClick={skipRound}
                      variant="outline"
                      size="lg"
                      className="flex-1 border-2"
                      disabled={gameState !== GAME_STATES.PLAYING}
                    >
                      <SkipForward className="mr-2 h-4 w-4" />
                      Skip
                    </Button>
                  </div>

                  {/* Error Message */}
                  {error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
                      <p className="font-medium">⚠️ {error}</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Predictions/Results Card */}
              <Card className="shadow-xl border-2 border-indigo-100">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-indigo-600" />
                    {gameState === GAME_STATES.PLAYING ? 'AI Guesses' : 'Result'}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {gameState === GAME_STATES.WON ? (
                    <div className="text-center space-y-6 py-8">
                      <CheckCircle2 className="w-24 h-24 mx-auto text-green-500" />
                      <div>
                        <h3 className="text-3xl font-bold text-green-600 mb-2">I got it!</h3>
                        <p className="text-xl text-gray-600">
                          It's a <span className="font-bold capitalize">{targetObject?.replace(/_/g, ' ')}</span>!
                        </p>
                        <p className="text-gray-500 mt-2">
                          Time taken: {timeTaken} seconds
                        </p>
                      </div>
                      <p className="text-sm text-gray-500">Starting next round...</p>
                    </div>
                  ) : gameState === GAME_STATES.LOST ? (
                    <div className="text-center space-y-6 py-8">
                      <XCircle className="w-24 h-24 mx-auto text-red-500" />
                      <div>
                        <h3 className="text-3xl font-bold text-red-600 mb-2">Time's up!</h3>
                        <p className="text-xl text-gray-600">
                          It was a <span className="font-bold capitalize">{targetObject?.replace(/_/g, ' ')}</span>
                        </p>
                        {predictions.length > 0 && (
                          <p className="text-gray-500 mt-2">
                            I thought it was: <span className="font-semibold capitalize">{predictions[0].replace(/_/g, ' ')}</span>
                          </p>
                        )}
                      </div>
                      <Button
                        onClick={nextRound}
                        size="lg"
                        className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
                      >
                        Next Round
                      </Button>
                    </div>
                  ) : predictions.length === 0 ? (
                    <div className="h-64 flex items-center justify-center text-muted-foreground">
                      <div className="text-center">
                        <Wand2 className="w-16 h-16 mx-auto mb-4 text-purple-300" />
                        <p>Start drawing and I'll try to guess!</p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {loading && (
                        <div className="flex items-center justify-center gap-2 p-3 bg-purple-50 border border-purple-200 rounded-lg mb-4">
                          <Loader2 className="h-4 w-4 animate-spin text-purple-600" />
                          <span className="text-sm text-purple-700">Thinking...</span>
                        </div>
                      )}
                      {predictions.map((prediction, index) => (
                        <div
                          key={index}
                          className="flex items-center gap-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border-2 border-purple-200 transition-all duration-200"
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
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6 mt-16">
        <div className="container mx-auto px-4 text-center text-sm">
          <p className="opacity-80">
            Inspired by Google's Quick, Draw! • Model: MobileNetV1 • Built with React & TensorFlow
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
