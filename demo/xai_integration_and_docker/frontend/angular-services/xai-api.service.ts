import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ModelInfo {
  model_loaded: boolean;
  model_type?: string;
  device?: string;
  parameters?: number;
}

export interface DatasetInfo {
  dataset_loaded: boolean;
  dataset_shape?: number[];
  data_type?: string;
  device?: string;
}

export interface XAIInput {
  data: number[][];
  shape: number[];
  device?: string;
  requires_grad?: boolean;
}

export interface XAIResponse {
  message: string;
  sample: { [key: string]: number };
  feature_names: string[];
  input_shape: number[];
}

export interface LIMEResponse extends XAIResponse {
  lime_values: { [key: string]: number };
  contribution: Array<{ feature: string; value: number }>;
  explanation_available: boolean;
}

export interface SHAPResponse extends XAIResponse {
  shap_values: { [key: string]: number };
  contribution: Array<{ feature: string; value: number }>;
}

@Injectable({
  providedIn: 'root'
})
export class XAIApiService {
  private readonly baseUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) {}

  // Model Management
  getModelInfo(): Observable<ModelInfo> {
    return this.http.get<ModelInfo>(`${this.baseUrl}/model`);
  }

  uploadModel(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/model`, formData);
  }

  // Dataset Management
  getDatasetInfo(): Observable<DatasetInfo> {
    return this.http.get<DatasetInfo>(`${this.baseUrl}/dataset`);
  }

  uploadDataset(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/dataset`, formData);
  }

  // LIME Analysis
  calculateLIME(data: XAIInput): Observable<LIMEResponse> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<LIMEResponse>(`${this.baseUrl}/lime/calculate`, data, { headers });
  }

  // SHAP Analysis
  calculateSHAP(data: XAIInput): Observable<SHAPResponse> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<SHAPResponse>(`${this.baseUrl}/shap/calculate`, data, { headers });
  }

  // Instance Management
  uploadInstanceFile(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/dataset`, formData);
  }

  // LIME Analysis with String JSON
  calculateLIMEString(data: XAIInput): Observable<LIMEResponse> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<LIMEResponse>(`${this.baseUrl}/lime/calculate/string_json`, data, { headers });
  }

  // SHAP Analysis with String JSON
  calculateSHAPString(data: XAIInput): Observable<SHAPResponse> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<SHAPResponse>(`${this.baseUrl}/shap/calculate/string_json`, data, { headers });
  }

  // Health Check
  healthCheck(): Observable<any> {
    return this.http.get(`${this.baseUrl}/model`);
  }
}
