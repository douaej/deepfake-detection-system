export type DetectionResult = 'REAL' | 'FAKE' | 'UNKNOWN'

export interface Evidence {
  num_faces?: number
  texture?: {
    homogeneity: number
  }
  heatmap_base64?: string
  message?: string
  [key: string]: any
}

export interface ImageResult {
  result: DetectionResult
  confidence: number
  fake_probability: number
  processing_time_ms: number
  evidence: Evidence
  explanation?: string
  request_id?: string
}

export interface VideoResult {
  overall_result: DetectionResult
  overall_prediction?: 0 | 1 // Raw API response
  avg_confidence: number
  stats: {
    fake_ratio?: number
    num_frames?: number
    mean_confidence?: number
    std_confidence?: number
    min_confidence?: number
    max_confidence?: number
    fake_count?: number
  }
  processing_time_ms: number
  request_id?: string
}

export type AnalysisResult = ImageResult | VideoResult

export function isImageResult(result: AnalysisResult): result is ImageResult {
  return 'fake_probability' in result
}

export function isVideoResult(result: AnalysisResult): result is VideoResult {
  return 'overall_result' in result && 'stats' in result
}