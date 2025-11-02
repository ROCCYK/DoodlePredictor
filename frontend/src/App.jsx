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

// Czech translations for all categories
const CZECH_TRANSLATIONS = {
  'airplane': 'letadlo',
  'alarm clock': 'budík',
  'ambulance': 'sanitka',
  'angel': 'anděl',
  'animal migration': 'migrace zvířat',
  'ant': 'mravenec',
  'anvil': 'kovadlina',
  'apple': 'jablko',
  'arm': 'paže',
  'asparagus': 'chřest',
  'axe': 'sekera',
  'backpack': 'batoh',
  'banana': 'banán',
  'bandage': 'obvaz',
  'barn': 'stodola',
  'baseball': 'baseball',
  'baseball bat': 'baseballová pálka',
  'basket': 'košík',
  'basketball': 'basketbal',
  'bat': 'netopýr',
  'bathtub': 'vana',
  'beach': 'pláž',
  'bear': 'medvěd',
  'beard': 'vousy',
  'bed': 'postel',
  'bee': 'včela',
  'belt': 'opasek',
  'bench': 'lavička',
  'bicycle': 'kolo',
  'binoculars': 'dalekohled',
  'bird': 'pták',
  'birthday cake': 'narozeninový dort',
  'blackberry': 'ostružina',
  'blueberry': 'borůvka',
  'book': 'kniha',
  'boomerang': 'bumerang',
  'bottlecap': 'uzávěr',
  'bowtie': 'motýlek',
  'bracelet': 'náramek',
  'brain': 'mozek',
  'bread': 'chléb',
  'bridge': 'most',
  'broccoli': 'brokolice',
  'broom': 'koště',
  'bucket': 'kbelík',
  'bulldozer': 'buldozer',
  'bus': 'autobus',
  'bush': 'keř',
  'butterfly': 'motýl',
  'cactus': 'kaktus',
  'cake': 'dort',
  'calculator': 'kalkulačka',
  'calendar': 'kalendář',
  'camel': 'velbloud',
  'camera': 'fotoaparát',
  'camouflage': 'maskování',
  'campfire': 'táborák',
  'candle': 'svíčka',
  'cannon': 'dělo',
  'canoe': 'kánoe',
  'car': 'auto',
  'carrot': 'mrkev',
  'castle': 'hrad',
  'cat': 'kočka',
  'ceiling fan': 'stropní ventilátor',
  'cell phone': 'mobilní telefon',
  'cello': 'violoncello',
  'chair': 'židle',
  'chandelier': 'lustr',
  'church': 'kostel',
  'circle': 'kruh',
  'clarinet': 'klarinet',
  'clock': 'hodiny',
  'cloud': 'mrak',
  'coffee cup': 'hrnek na kávu',
  'compass': 'kompas',
  'computer': 'počítač',
  'cookie': 'sušenka',
  'cooler': 'chladící box',
  'couch': 'pohovka',
  'cow': 'kráva',
  'crab': 'krab',
  'crayon': 'pastelka',
  'crocodile': 'krokodýl',
  'crown': 'koruna',
  'cruise ship': 'výletní loď',
  'cup': 'šálek',
  'diamond': 'diamant',
  'dishwasher': 'myčka',
  'diving board': 'skokanské prkno',
  'dog': 'pes',
  'dolphin': 'delfín',
  'donut': 'kobliha',
  'door': 'dveře',
  'dragon': 'drak',
  'dresser': 'komoda',
  'drill': 'vrtačka',
  'drums': 'bubny',
  'duck': 'kachna',
  'dumbbell': 'činky',
  'ear': 'ucho',
  'elbow': 'loket',
  'elephant': 'slon',
  'envelope': 'obálka',
  'eraser': 'guma',
  'eye': 'oko',
  'eyeglasses': 'brýle',
  'face': 'obličej',
  'fan': 'ventilátor',
  'feather': 'pírko',
  'fence': 'plot',
  'finger': 'prst',
  'fire hydrant': 'požární hydrant',
  'fireplace': 'krb',
  'firetruck': 'hasičské auto',
  'fish': 'ryba',
  'flamingo': 'plameňák',
  'flashlight': 'baterka',
  'flip flops': 'žabky',
  'floor lamp': 'lampa',
  'flower': 'květina',
  'flying saucer': 'létající talíř',
  'foot': 'noha',
  'fork': 'vidlička',
  'frog': 'žába',
  'frying pan': 'pánev',
  'garden': 'zahrada',
  'garden hose': 'zahradní hadice',
  'giraffe': 'žirafa',
  'goatee': 'bradka',
  'golf club': 'golfová hůl',
  'grapes': 'hrozny',
  'grass': 'tráva',
  'guitar': 'kytara',
  'hamburger': 'hamburger',
  'hammer': 'kladivo',
  'hand': 'ruka',
  'harp': 'harfa',
  'hat': 'klobouk',
  'headphones': 'sluchátka',
  'hedgehog': 'ježek',
  'helicopter': 'helikoptéra',
  'helmet': 'helma',
  'hexagon': 'šestiúhelník',
  'hockey puck': 'hokejový puk',
  'hockey stick': 'hokejka',
  'horse': 'kůň',
  'hospital': 'nemocnice',
  'hot air balloon': 'horkovzdušný balón',
  'hot dog': 'párek v rohlíku',
  'hot tub': 'vířivka',
  'hourglass': 'přesýpací hodiny',
  'house': 'dům',
  'house plant': 'pokojová rostlina',
  'hurricane': 'hurikán',
  'ice cream': 'zmrzlina',
  'jacket': 'bunda',
  'jail': 'vězení',
  'kangaroo': 'klokan',
  'key': 'klíč',
  'keyboard': 'klávesnice',
  'knee': 'koleno',
  'ladder': 'žebřík',
  'lantern': 'lucerna',
  'laptop': 'notebook',
  'leaf': 'list',
  'leg': 'noha',
  'light bulb': 'žárovka',
  'lighthouse': 'maják',
  'lightning': 'blesk',
  'line': 'čára',
  'lion': 'lev',
  'lipstick': 'rtěnka',
  'lobster': 'humr',
  'lollipop': 'lízátko',
  'mailbox': 'poštovní schránka',
  'map': 'mapa',
  'marker': 'fix',
  'matches': 'sirky',
  'megaphone': 'megafon',
  'mermaid': 'mořská panna',
  'microphone': 'mikrofon',
  'microwave': 'mikrovlnka',
  'monkey': 'opice',
  'moon': 'měsíc',
  'mosquito': 'komár',
  'motorbike': 'motorka',
  'mountain': 'hora',
  'mouse': 'myš',
  'moustache': 'knír',
  'mouth': 'ústa',
  'mug': 'hrnek',
  'mushroom': 'houba',
  'nail': 'hřebík',
  'necklace': 'náhrdelník',
  'nose': 'nos',
  'ocean': 'oceán',
  'octagon': 'osmiúhelník',
  'octopus': 'chobotnice',
  'onion': 'cibule',
  'oven': 'trouba',
  'owl': 'sova',
  'paint can': 'plechovka s barvou',
  'paintbrush': 'štětec',
  'palm tree': 'palma',
  'panda': 'panda',
  'pants': 'kalhoty',
  'paper clip': 'sponka',
  'parachute': 'padák',
  'parrot': 'papoušek',
  'passport': 'pas',
  'peanut': 'burák',
  'pear': 'hruška',
  'peas': 'hrášek',
  'pencil': 'tužka',
  'penguin': 'tučňák',
  'piano': 'piano',
  'pickup truck': 'pickup',
  'picture frame': 'rámeček',
  'pig': 'prase',
  'pillow': 'polštář',
  'pineapple': 'ananas',
  'pizza': 'pizza',
  'pliers': 'kleště',
  'police car': 'policejní auto',
  'pond': 'rybník',
  'pool': 'bazén',
  'popsicle': 'nanuková zmrzlina',
  'postcard': 'pohlednice',
  'potato': 'brambor',
  'power outlet': 'zásuvka',
  'purse': 'kabelka',
  'rabbit': 'králík',
  'raccoon': 'mýval',
  'radio': 'rádio',
  'rain': 'déšť',
  'rainbow': 'duha',
  'rake': 'hrábě',
  'remote control': 'dálkový ovladač',
  'rhinoceros': 'nosorožec',
  'river': 'řeka',
  'roller coaster': 'horská dráha',
  'rollerskates': 'kolečkové brusle',
  'sailboat': 'plachetnice',
  'sandwich': 'sendvič',
  'saw': 'pila',
  'saxophone': 'saxofon',
  'school bus': 'školní autobus',
  'scissors': 'nůžky',
  'scorpion': 'štír',
  'screwdriver': 'šroubovák',
  'sea turtle': 'mořská želva',
  'see saw': 'houpačka',
  'shark': 'žralok',
  'sheep': 'ovce',
  'shoe': 'bota',
  'shorts': 'kraťasy',
  'shovel': 'lopata',
  'sink': 'dřez',
  'skateboard': 'skateboard',
  'skull': 'lebka',
  'skyscraper': 'mrakodrap',
  'sleeping bag': 'spacák',
  'smiley face': 'smajlík',
  'snail': 'šnek',
  'snake': 'had',
  'snorkel': 'šnorchl',
  'snowflake': 'sněhová vločka',
  'snowman': 'sněhulák',
  'soccer ball': 'fotbalový míč',
  'sock': 'ponožka',
  'speedboat': 'rychlý člun',
  'spider': 'pavouk',
  'spoon': 'lžíce',
  'spreadsheet': 'tabulka',
  'square': 'čtverec',
  'squiggle': 'čmáranice',
  'squirrel': 'veverka',
  'stairs': 'schody',
  'star': 'hvězda',
  'steak': 'biftek',
  'stereo': 'stereo',
  'stethoscope': 'stetoskop',
  'stitches': 'stehy',
  'stop sign': 'značka stop',
  'stove': 'sporák',
  'strawberry': 'jahoda',
  'streetlight': 'pouliční lampa',
  'string bean': 'fazole',
  'submarine': 'ponorka',
  'suitcase': 'kufr',
  'sun': 'slunce',
  'swan': 'labuť',
  'sweater': 'svetr',
  'swing set': 'houpačka',
  'sword': 'meč',
  't-shirt': 'tričko',
  'table': 'stůl',
  'teapot': 'čajová konvice',
  'teddy-bear': 'plyšový medvídek',
  'telephone': 'telefon',
  'television': 'televize',
  'tennis racquet': 'tenisová raketa',
  'tent': 'stan',
  'The Eiffel Tower': 'Eiffelova věž',
  'The Great Wall of China': 'Velká čínská zeď',
  'The Mona Lisa': 'Mona Lisa',
  'tiger': 'tygr',
  'toaster': 'topinkovač',
  'toe': 'palec u nohy',
  'toilet': 'záchod',
  'tooth': 'zub',
  'toothbrush': 'kartáček na zuby',
  'toothpaste': 'zubní pasta',
  'tornado': 'tornádo',
  'tractor': 'traktor',
  'traffic light': 'semafor',
  'train': 'vlak',
  'tree': 'strom',
  'triangle': 'trojúhelník',
  'trombone': 'pozoun',
  'truck': 'nákladní auto',
  'trumpet': 'trumpeta',
  'umbrella': 'deštník',
  'underwear': 'spodní prádlo',
  'van': 'dodávka',
  'vase': 'váza',
  'violin': 'housle',
  'washing machine': 'pračka',
  'watermelon': 'meloun',
  'waterslide': 'vodní skluzavka',
  'whale': 'velryba',
  'wheel': 'kolo',
  'windmill': 'větrný mlýn',
  'wine bottle': 'láhev vína',
  'wine glass': 'sklenice na víno',
  'wristwatch': 'náramkové hodinky',
  'yoga': 'jóga',
  'zebra': 'zebra',
  'zigzag': 'cik cak'
}

function App() {
  const sigCanvas = useRef(null)

  // Game state
  const [gameState, setGameState] = useState(GAME_STATES.IDLE)
  const [targetObject, setTargetObject] = useState(null)
  const [targetObjectCzech, setTargetObjectCzech] = useState(null)
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
  const lastSpokenPredictionRef = useRef(null)
  const speechSynthesisRef = useRef(null)

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

  // Speech synthesis function with Czech language
  const speak = (text) => {
    // Cancel any ongoing speech
    if (speechSynthesisRef.current) {
      window.speechSynthesis.cancel()
    }

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'cs-CZ' // Czech language
    utterance.rate = 1.0
    utterance.pitch = 1.0
    utterance.volume = 1.0

    speechSynthesisRef.current = utterance
    window.speechSynthesis.speak(utterance)
  }

  const speakPredictions = (predictionsList) => {
    if (predictionsList.length === 0) return

    // Only speak if the top prediction has changed
    const topPrediction = predictionsList[0]
    if (lastSpokenPredictionRef.current === topPrediction) return

    lastSpokenPredictionRef.current = topPrediction

    // Get Czech translation
    const czechName = CZECH_TRANSLATIONS[topPrediction] || topPrediction.replace(/_/g, ' ')

    // Czech phrase patterns
    const patterns = [
      `Vidím ${czechName}`,
      `Je to ${czechName}?`,
      `Možná ${czechName}`,
      `Nebo ${czechName}`,
      `${czechName}?`
    ]

    const randomPattern = patterns[Math.floor(Math.random() * patterns.length)]
    speak(randomPattern)
  }

  const getRandomCategory = () => {
    if (categories.length === 0) return null
    const randomIndex = Math.floor(Math.random() * categories.length)
    return categories[randomIndex]
  }

  const startNewRound = () => {
    const newTarget = getRandomCategory()
    if (!newTarget) return

    // Get Czech translation
    const czechTarget = CZECH_TRANSLATIONS[newTarget] || newTarget.replace(/_/g, ' ')

    // Reset everything for new round
    if (sigCanvas.current) {
      sigCanvas.current.clear()
    }
    setPredictions([])
    setError(null)
    setTargetObject(newTarget)
    setTargetObjectCzech(czechTarget)
    setTimeLeft(20)
    setGameState(GAME_STATES.PLAYING)
    hasStartedDrawingRef.current = false
    roundStartTimeRef.current = Date.now()
    lastSpokenPredictionRef.current = null

    // Cancel any ongoing speech
    window.speechSynthesis.cancel()

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
    speak("Promiň, nepodařilo se mi to uhodnout")
  }

  const handleWin = () => {
    const endTime = Date.now()
    const timeTakenSeconds = Math.round((endTime - roundStartTimeRef.current) / 1000)
    setTimeTaken(timeTakenSeconds)
    setGameState(GAME_STATES.WON)
    setScore(prev => prev + 1)
    clearAllTimers()

    // Cancel any ongoing speech first
    window.speechSynthesis.cancel()

    // Delay the win speech slightly to ensure previous speech is cancelled
    setTimeout(() => {
      speak(`Už vím, je to ${targetObjectCzech}!`)
    }, 200)
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
        throw new Error('Predikce selhala')
      }

      const data = await response.json()
      setPredictions(data.predictions)

      // Speak the predictions
      speakPredictions(data.predictions)

      // Check if target is in predictions (case insensitive, handle underscores)
      const normalizedTarget = targetObject.toLowerCase().replace(/ /g, '_')
      const normalizedPredictions = data.predictions.map(p => p.toLowerCase())

      if (normalizedPredictions.includes(normalizedTarget)) {
        handleWin()
      }
    } catch (err) {
      setError(err.message || 'Došlo k chybě')
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

  // Convert English prediction to Czech for display
  const getCzechPrediction = (englishPrediction) => {
    return CZECH_TRANSLATIONS[englishPrediction] || englishPrediction.replace(/_/g, ' ')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Sparkles className="w-8 h-8" />
              <h1 className="text-3xl font-bold">Rychle, kresli!</h1>
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
              <h2 className="text-4xl font-bold text-gray-800">Rychle, kresli!</h2>
              <p className="text-xl text-gray-600">
                Umíš kreslit dost rychle, aby ti umělá inteligence rozpoznala tvé kresby?
              </p>
              <div className="space-y-2 text-left bg-purple-50 p-6 rounded-lg">
                <h3 className="font-bold text-lg mb-3">Jak hrát:</h3>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">1.</span>
                  Uvidíš, co máš nakreslit
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">2.</span>
                  Kresli co nejrychleji!
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">3.</span>
                  AI se bude snažit uhodnout, co to je
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-bold text-purple-600">4.</span>
                  Máš 20 sekund na každou kresbu
                </p>
              </div>
              <Button
                onClick={startGame}
                size="lg"
                className="text-xl px-12 py-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
              >
                <Play className="mr-2 h-6 w-6" />
                Pojďme kreslit!
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
                      <p className="text-sm text-gray-500 uppercase tracking-wide">Nakresli:</p>
                      <p className="text-4xl font-bold text-gray-800">
                        {targetObjectCzech}
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
                    Kresli zde
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
                      Vymazat
                    </Button>
                    <Button
                      onClick={skipRound}
                      variant="outline"
                      size="lg"
                      className="flex-1 border-2"
                      disabled={gameState !== GAME_STATES.PLAYING}
                    >
                      <SkipForward className="mr-2 h-4 w-4" />
                      Přeskočit
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
                    {gameState === GAME_STATES.PLAYING ? 'Tipy AI' : 'Výsledek'}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {gameState === GAME_STATES.WON ? (
                    <div className="text-center space-y-6 py-8">
                      <CheckCircle2 className="w-24 h-24 mx-auto text-green-500" />
                      <div>
                        <h3 className="text-3xl font-bold text-green-600 mb-2">Mám to!</h3>
                        <p className="text-xl text-gray-600">
                          Je to <span className="font-bold">{targetObjectCzech}</span>!
                        </p>
                        <p className="text-gray-500 mt-2">
                          Čas: {timeTaken} sekund
                        </p>
                      </div>
                      <p className="text-sm text-gray-500">Další kolo začíná...</p>
                    </div>
                  ) : gameState === GAME_STATES.LOST ? (
                    <div className="text-center space-y-6 py-8">
                      <XCircle className="w-24 h-24 mx-auto text-red-500" />
                      <div>
                        <h3 className="text-3xl font-bold text-red-600 mb-2">Čas vypršel!</h3>
                        <p className="text-xl text-gray-600">
                          Bylo to <span className="font-bold">{targetObjectCzech}</span>
                        </p>
                        {predictions.length > 0 && (
                          <p className="text-gray-500 mt-2">
                            Myslel jsem, že je to: <span className="font-semibold">{getCzechPrediction(predictions[0])}</span>
                          </p>
                        )}
                      </div>
                      <Button
                        onClick={nextRound}
                        size="lg"
                        className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
                      >
                        Další kolo
                      </Button>
                    </div>
                  ) : predictions.length === 0 ? (
                    <div className="h-64 flex items-center justify-center text-muted-foreground">
                      <div className="text-center">
                        <Wand2 className="w-16 h-16 mx-auto mb-4 text-purple-300" />
                        <p>Začni kreslit a já se pokusím uhodnout!</p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {loading && (
                        <div className="flex items-center justify-center gap-2 p-3 bg-purple-50 border border-purple-200 rounded-lg mb-4">
                          <Loader2 className="h-4 w-4 animate-spin text-purple-600" />
                          <span className="text-sm text-purple-700">Přemýšlím...</span>
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
                          <span className="text-xl font-semibold text-gray-800 flex-1">
                            {getCzechPrediction(prediction)}
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
            Inspirováno hrou Quick, Draw! od Google • Model: MobileNetV1 • Vytvořeno pomocí React & TensorFlow
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
