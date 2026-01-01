import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Upload, Camera, AlertTriangle, Shield, 
  Activity, Search, Settings, Cpu, FileBox, Crosshair,
  Maximize2, Power, RefreshCw, Layers, History, Video
} from 'lucide-react'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs) { return twMerge(clsx(inputs)) }

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.45)
  const [systemTime, setSystemTime] = useState(new Date().toLocaleTimeString())
  const [modelInfo, setModelInfo] = useState({ name: "INITIALIZING", status: "checking" })
  
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    const timer = setInterval(() => setSystemTime(new Date().toLocaleTimeString()), 1000)
    checkBackend()
    return () => clearInterval(timer)
  }, [])

  const checkBackend = async () => {
    try {
      const res = await axios.get('http://localhost:8002/')
      setModelInfo({ name: res.data.model.split(/[\\/]/).pop(), status: "online" })
    } catch (err) {
      setModelInfo({ name: "CONNECTION LOST", status: "offline" })
    }
  }

  const [mode, setMode] = useState('image') // 'image' | 'video'
  const [videoSource, setVideoSource] = useState('webcam') // 'webcam' | 'file'
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const videoFileRef = useRef(null)
  const videoCanvasRef = useRef(null)

  useEffect(() => {
    return () => {
      stopVideo()
    }
  }, [])

  const startWebcam = async () => {
    stopVideo()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      videoRef.current.srcObject = stream
      videoRef.current.play()
      streamRef.current = stream
      setLoading(true)
      analyzeVideoLoop()
    } catch (err) {
      console.error(err)
      alert("Kamera erişimi reddedildi!")
    }
  }

  const handleVideoUpload = (file) => {
    if (!file) return
    stopVideo()
    const url = URL.createObjectURL(file)
    videoRef.current.src = url
    videoRef.current.srcObject = null
    // Video yüklendiğinde otomatik başlatmıyoruz, kullanıcı başlatmalı
  }

  const toggleVideoPlay = () => {
    if (videoRef.current.paused) {
        videoRef.current.play()
        setLoading(true)
        analyzeVideoLoop()
    } else {
        videoRef.current.pause()
        setLoading(false)
    }
  }

  const stopVideo = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
        videoRef.current.pause()
        if (videoRef.current.src) {
            URL.revokeObjectURL(videoRef.current.src)
            videoRef.current.src = ""
        }
        videoRef.current.srcObject = null
    }
    setLoading(false)
  }

  const analyzeVideoLoop = async () => {
    if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
        setLoading(false)
        return
    }

    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth || 640
    canvas.height = videoRef.current.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    
    canvas.toBlob(async (blob) => {
      if (!blob) return
      const formData = new FormData()
      formData.append('file', blob, 'frame.jpg')

      try {
        const response = await axios.post(
          `http://localhost:8002/detect?conf=${conf}&iou=${iou}`, 
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        setResults(response.data)
        
        // Video üzerine çizim yap
        if(videoCanvasRef.current) {
            const vCanvas = videoCanvasRef.current;
            vCanvas.width = canvas.width;
            vCanvas.height = canvas.height;
            const vCtx = vCanvas.getContext('2d');
            vCtx.clearRect(0, 0, vCanvas.width, vCanvas.height);
            
            // Mevcut çizim fonksiyonunu yeniden kullan
            drawDetectionsOnContext(vCtx, response.data.detections);
        }

      } catch (err) {
        console.error(err)
      } finally {
        // Döngü devam ediyor
        if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
            requestAnimationFrame(analyzeVideoLoop) 
        } else {
            setLoading(false)
        }
      }
    }, 'image/jpeg', 0.8) // Kaliteyi biraz düşürelim hız için
  }

  // Ortak çizim fonksiyonu (Context alır)
  const drawDetectionsOnContext = (ctx, detections) => {
      detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox
        const width = x2 - x1
        const height = y2 - y1
        const isThreat = ['Gun', 'Knife', 'Bullet'].includes(det.class)
        const color = isThreat ? '#ef4444' : '#0ea5e9' // Red vs Sky Blue
        
        // Target Box Style
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        
        // Corners only drawing
        const lineLen = Math.min(width, height) * 0.2
        ctx.beginPath()
        
        // TL
        ctx.moveTo(x1, y1 + lineLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + lineLen, y1)
        // TR
        ctx.moveTo(x2 - lineLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + lineLen)
        // BR
        ctx.moveTo(x2, y2 - lineLen); ctx.lineTo(x2, y2); ctx.lineTo(x2 - lineLen, y2)
        // BL
        ctx.moveTo(x1 + lineLen, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - lineLen)
        
        ctx.stroke()
        
        // Semi-transparent fill
        ctx.fillStyle = color
        ctx.globalAlpha = 0.1
        ctx.fillRect(x1, y1, width, height)
        ctx.globalAlpha = 1.0

        // Label Tag
        const text = `${det.class} ${(det.confidence * 100).toFixed(0)}%`
        ctx.font = '500 14px "JetBrains Mono", monospace'
        const metrics = ctx.measureText(text)
        const pad = 6
        
        // Label Background
        ctx.fillStyle = color
        ctx.fillRect(x1, y1 - 24, metrics.width + pad * 2, 24)
        
        // Label Text
        ctx.fillStyle = isThreat ? '#fff' : '#000'
        ctx.fillText(text, x1 + pad, y1 - 7)
      })
  }

  useEffect(() => {
    if (mode === 'image') stopVideo()
  }, [mode])

  const handleFile = (file) => {
    if (!file) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResults(null)
  }

  const handleAnalyze = async () => {
    if (!image) return
    setLoading(true)
    
    const formData = new FormData()
    formData.append('file', image)
    
    try {
      const response = await axios.post(
        `http://localhost:8002/detect?conf=${conf}&iou=${iou}`, 
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      
      setResults(response.data)
      setTimeout(() => drawDetections(response.data.detections), 100)
      
    } catch (err) {
      console.error(err)
      alert("Analysis failed. Check backend connection.")
    } finally {
      setLoading(false)
    }
  }

  const drawDetections = (detections) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const img = new Image()
    
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      drawDetectionsOnContext(ctx, detections)
    }
    img.src = preview
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-300 font-sans overflow-hidden selection:bg-primary/30">
      
      {/* LEFT SIDEBAR: Navigation & Status */}
      <aside className="w-20 lg:w-64 bg-slate-900 border-r border-slate-800 flex flex-col z-20">
        <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-slate-800">
          <Shield className="w-8 h-8 text-primary animate-pulse" />
          <span className="hidden lg:block ml-3 font-bold text-lg text-slate-100 tracking-wider">X-RAY<span className="text-primary">VISION</span></span>
        </div>

        <nav className="flex-1 py-6 flex flex-col gap-2 px-3">
          <NavItem 
            icon={<Activity />} 
            label="Görüntü Analizi" 
            active={mode === 'image'} 
            onClick={() => setMode('image')} 
          />
          <NavItem 
            icon={<Video />} 
            label="Video Analizi" 
            active={mode === 'video'} 
            onClick={() => setMode('video')} 
          />
          <NavItem icon={<History />} label="Kayıtlar" />
          <NavItem icon={<Layers />} label="Modeller" />
          <NavItem icon={<Settings />} label="Sistem" />
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="glass-panel p-3 rounded-lg">
            <div className="flex items-center gap-3 mb-2">
              <Cpu className="w-4 h-4 text-slate-400" />
              <span className="hidden lg:block text-xs font-mono text-slate-400">MOTOR DURUMU</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={cn("w-2 h-2 rounded-full", modelInfo.status === 'online' ? "bg-emerald-500 shadow-neon-green" : "bg-red-500")} />
              <span className={cn("hidden lg:block text-xs font-bold truncate", modelInfo.status === 'online' ? "text-emerald-400" : "text-red-400")}>
                {modelInfo.status === 'online' ? 'ÇEVRİMİÇİ' : 'ÇEVRİMDIŞI'}
              </span>
            </div>
            <div className="hidden lg:block text-[10px] text-slate-500 mt-1 font-mono truncate">{modelInfo.name}</div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col min-w-0">
        
        {/* TOP BAR */}
        <header className="h-16 bg-slate-900/80 backdrop-blur border-b border-slate-800 flex items-center justify-between px-6 z-10">
          <div className="flex items-center gap-4">
            <span className="text-xs text-slate-400 font-mono flex items-center gap-2">
              <span className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              GERÇEK ZAMANLI ANALİZ
            </span>
          </div>
          <div className="font-mono text-xl font-bold text-slate-200 tracking-widest">
            {systemTime}
          </div>
        </header>

        <div className="flex-1 p-6 grid grid-cols-12 gap-6 overflow-hidden">
          
          {/* CENTER: Viewport */}
          <div className="col-span-12 lg:col-span-9 flex flex-col gap-4 min-h-0">
            
            {/* Main Display */}
            <div className="flex-1 glass-panel rounded-2xl relative overflow-hidden flex flex-col">
              {/* HUD Overlay */}
              <div className="absolute inset-0 pointer-events-none z-10 p-4 flex flex-col justify-between">
                <div className="flex justify-between">
                  <Crosshair className="w-6 h-6 text-slate-600/50" />
                  <Crosshair className="w-6 h-6 text-slate-600/50" />
                </div>
                <div className="flex justify-between">
                  <Crosshair className="w-6 h-6 text-slate-600/50" />
                  <Crosshair className="w-6 h-6 text-slate-600/50" />
                </div>
              </div>

              {/* Image/Video Area */}
              <div 
                className="flex-1 relative flex items-center justify-center bg-slate-950/50 overflow-hidden"
                onDragOver={(e) => mode === 'image' && e.preventDefault()}
                onDrop={(e) => { 
                  if(mode === 'image') {
                    e.preventDefault(); 
                    handleFile(e.dataTransfer.files[0]) 
                  }
                }}
              >
                {mode === 'image' ? (
                  preview ? (
                    <>
                      <canvas 
                        ref={canvasRef} 
                        className={cn(
                          "max-w-full max-h-full object-contain transition-all duration-500",
                          results ? "opacity-100 scale-100" : "opacity-0 scale-95 absolute"
                        )}
                      />
                      {!results && (
                        <img src={preview} alt="Scan Target" className="max-w-full max-h-full object-contain opacity-80" />
                      )}
                      
                      {/* Scanning Laser */}
                      {loading && <div className="scanner-line z-20" />}
                    </>
                  ) : (
                    <div className="text-center p-8 border border-dashed border-slate-700 rounded-xl hover:border-primary/50 hover:bg-primary/5 transition-all group cursor-pointer"
                        onClick={() => fileInputRef.current.click()}>
                      <Upload className="w-12 h-12 mx-auto text-slate-600 group-hover:text-primary mb-4 transition-colors" />
                      <h3 className="text-lg font-medium text-slate-300">Görüntü Kaynağı Yok</h3>
                      <p className="text-sm text-slate-500 mt-1 font-mono">X-RAY GÖRÜNTÜSÜ YÜKLE VEYA SÜRÜKLE</p>
                    </div>
                  )
                ) : (
                  /* Video Mode */
                  <div className="relative w-full h-full flex flex-col items-center justify-center">
                    
                    {/* Source Selector Overlay (if no video active) */}
                    {(!streamRef.current && (!videoRef.current?.src || videoRef.current.src === window.location.href)) && (
                       <div className="flex gap-4">
                           <div className="text-center p-8 border border-dashed border-slate-700 rounded-xl hover:border-primary/50 hover:bg-primary/5 transition-all group cursor-pointer w-64"
                               onClick={() => { setVideoSource('webcam'); startWebcam(); }}>
                            <Camera className="w-12 h-12 mx-auto text-slate-600 group-hover:text-primary mb-4 transition-colors" />
                            <h3 className="text-lg font-medium text-slate-300">Kamera Kullan</h3>
                            <p className="text-sm text-slate-500 mt-1 font-mono">CANLI KAMERA AKIŞI</p>
                          </div>

                          <div className="text-center p-8 border border-dashed border-slate-700 rounded-xl hover:border-primary/50 hover:bg-primary/5 transition-all group cursor-pointer w-64"
                               onClick={() => videoFileRef.current.click()}>
                            <Upload className="w-12 h-12 mx-auto text-slate-600 group-hover:text-primary mb-4 transition-colors" />
                            <h3 className="text-lg font-medium text-slate-300">Video Yükle</h3>
                            <p className="text-sm text-slate-500 mt-1 font-mono">MP4, WEBM, AVI</p>
                            <input 
                                type="file" 
                                ref={videoFileRef} 
                                className="hidden" 
                                accept="video/*" 
                                onChange={(e) => { setVideoSource('file'); handleVideoUpload(e.target.files[0]); }} 
                            />
                          </div>
                       </div>
                    )}

                    <video 
                      ref={videoRef} 
                      playsInline 
                      muted 
                      className={cn("max-w-full max-h-full object-contain", (!streamRef.current && (!videoRef.current?.src || videoRef.current.src === window.location.href)) ? "hidden" : "block")}
                    />
                    
                    {/* Video Overlay for Bounding Boxes */}
                    {(streamRef.current || (videoRef.current?.src && videoRef.current.src !== window.location.href)) && (
                        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                            <canvas 
                                ref={videoCanvasRef}
                                className="max-w-full max-h-full object-contain"
                            />
                        </div>
                    )}
                  </div>
                )}
                
                {loading && mode === 'image' && (
                  <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                    <div className="text-primary font-mono text-lg tracking-[0.2em] animate-pulse border border-primary px-4 py-2 bg-primary/10">
                      İŞLENİYOR...
                    </div>
                  </div>
                )}
              </div>

              {/* Bottom Info Bar */}
              <div className="h-12 bg-slate-900 border-t border-slate-800 flex items-center px-4 justify-between text-xs font-mono text-slate-400">
                <div className="flex gap-4">
                  <span>MOD: {mode === 'image' ? 'GÖRÜNTÜ' : 'VİDEO/KAMERA'}</span>
                  <span>DURUM: {loading ? 'AKTİF' : 'BEKLEMEDE'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={cn("w-2 h-2 rounded-full", (preview || streamRef.current) ? "bg-primary" : "bg-slate-700")} />
                  {(preview || streamRef.current) ? 'KAYNAK BAĞLI' : 'KAYNAK YOK'}
                </div>
              </div>
            </div>

            {/* Controls */}
            {mode === 'image' ? (
              <div className="grid grid-cols-4 gap-4">
                <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={(e) => handleFile(e.target.files[0])} />
                
                <button onClick={() => fileInputRef.current.click()} className="col-span-1 glass-panel hover:bg-slate-800 transition-all rounded-xl p-4 flex flex-col items-center justify-center gap-2 group">
                  <FileBox className="w-6 h-6 text-slate-400 group-hover:text-white" />
                  <span className="text-xs font-bold text-slate-400 group-hover:text-white">YÜKLE</span>
                </button>
                
                <button className="col-span-1 glass-panel hover:bg-slate-800 transition-all rounded-xl p-4 flex flex-col items-center justify-center gap-2 group" onClick={checkBackend}>
                  <RefreshCw className="w-6 h-6 text-slate-400 group-hover:text-white" />
                  <span className="text-xs font-bold text-slate-400 group-hover:text-white">SIFIRLA</span>
                </button>

                <button 
                  onClick={handleAnalyze}
                  disabled={!image || loading}
                  className={cn(
                    "col-span-2 rounded-xl p-4 flex items-center justify-center gap-3 transition-all font-bold tracking-wider shadow-lg transform hover:scale-[1.02] active:scale-95",
                    !image || loading
                      ? "bg-slate-800 text-slate-600 border border-slate-700 cursor-not-allowed"
                      : "bg-gradient-to-r from-primary to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white shadow-neon border border-primary/50"
                  )}
                >
                  {loading ? (
                    <RefreshCw className="w-6 h-6 animate-spin" />
                  ) : (
                    <Search className="w-6 h-6" />
                  )}
                  <span className="text-lg">TARAMAYI BAŞLAT</span>
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                 {(streamRef.current || (videoRef.current?.src && videoRef.current.src !== window.location.href)) ? (
                    <>
                        <button 
                        onClick={stopVideo}
                        className="col-span-1 rounded-xl p-4 flex items-center justify-center gap-3 transition-all font-bold tracking-wider shadow-lg bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700"
                        >
                        <RefreshCw className="w-6 h-6" />
                        <span className="text-lg">KAPAT</span>
                        </button>

                        <button 
                        onClick={streamRef.current ? () => {} : toggleVideoPlay}
                        className={cn(
                            "col-span-1 rounded-xl p-4 flex items-center justify-center gap-3 transition-all font-bold tracking-wider shadow-lg transform hover:scale-[1.02] active:scale-95",
                            loading
                            ? "bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30"
                            : "bg-primary/20 text-primary border border-primary/50 hover:bg-primary/30"
                        )}
                        >
                        <Power className="w-6 h-6" />
                        <span className="text-lg">
                            {streamRef.current ? 'CANLI AKIŞ' : (loading ? 'DURDUR' : 'OYNAT & TARA')}
                        </span>
                        </button>
                    </>
                 ) : (
                    <div className="col-span-2 text-center text-slate-500 text-sm font-mono p-4">
                        KAYNAK SEÇİNİZ
                    </div>
                 )}
              </div>
            )}
          </div>

          {/* RIGHT: Analysis Data */}
          <div className="col-span-12 lg:col-span-3 flex flex-col gap-6 min-h-0">
            
            {/* Detection List */}
            <div className="flex-1 glass-panel rounded-2xl flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                <h3 className="font-bold text-sm tracking-wider text-slate-200 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-primary" /> TEHDİT AKIŞI
                </h3>
                {results && <span className="bg-primary/20 text-primary text-xs px-2 py-0.5 rounded font-mono">{results.count} TESPİT</span>}
              </div>
              
              <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
                <AnimatePresence>
                  {results?.detections.map((det, idx) => (
                    <motion.div 
                      key={idx}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className={cn(
                        "p-3 rounded border flex justify-between items-center group relative overflow-hidden",
                        ['Gun', 'Knife'].includes(det.class) 
                          ? "bg-red-500/10 border-red-500/30 text-red-200" 
                          : "bg-slate-800/50 border-slate-700 text-slate-300"
                      )}
                    >
                      {/* Threat Indicator */}
                      {['Gun', 'Knife'].includes(det.class) && (
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-red-500 animate-pulse" />
                      )}
                      
                      <div className="flex flex-col z-10 ml-2">
                        <span className="font-bold text-sm">{det.class}</span>
                        <span className="text-[10px] opacity-70 font-mono">{det.bbox.join(', ')}</span>
                      </div>
                      
                      <div className="z-10 flex flex-col items-end">
                        <span className="font-mono font-bold text-lg leading-none">{(det.confidence * 100).toFixed(0)}<span className="text-[10px]">%</span></span>
                        {['Gun', 'Knife'].includes(det.class) && <AlertTriangle className="w-3 h-3 text-red-500 mt-1" />}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                
                {!results && (
                  <div className="h-full flex flex-col items-center justify-center text-slate-700 gap-2">
                    <Maximize2 className="w-8 h-8 opacity-20" />
                    <span className="text-xs font-mono">SİSTEM BOŞTA</span>
                  </div>
                )}
              </div>
            </div>

            {/* Threshold Config */}
            <div className="glass-panel rounded-2xl p-5">
              <h3 className="font-bold text-xs text-slate-400 uppercase mb-4 flex items-center gap-2">
                <Settings className="w-3 h-3" /> Parametre Ayarları
              </h3>
              
              <div className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">GÜVEN EŞİĞİ</span>
                    <span className="text-primary font-mono">{conf}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.05" value={conf} onChange={e => setConf(e.target.value)} className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-neon" />
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">IOU EŞİĞİ</span>
                    <span className="text-primary font-mono">{iou}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.05" value={iou} onChange={e => setIou(e.target.value)} className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-neon" />
                </div>
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  )
}

const NavItem = ({ icon, label, active, onClick }) => (
  <button 
    onClick={onClick}
    className={cn(
    "flex lg:justify-start justify-center items-center gap-3 px-3 py-3 rounded-lg transition-all group w-full",
    active 
      ? "bg-primary/10 text-primary border border-primary/20" 
      : "text-slate-500 hover:text-slate-200 hover:bg-slate-800"
  )}>
    {icon}
    <span className="hidden lg:block text-sm font-medium">{label}</span>
  </button>
)

export default App
