import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class DataApiService {

  // Backend consolidated under port 5000 and prefix /api
  readonly endpointDatasetAPI = "http://127.0.0.1:5000/api/dataset"
  readonly endpointModelAPI = "http://127.0.0.1:5000/api/model"

  constructor(private http:HttpClient) { }

  uploadDataset(file:File){
    const formData = new FormData();
    formData.append('file', file);  
    return this.http.post(`${this.endpointDatasetAPI}`, formData)
  }

  listDatasets(){
    return this.http.get(`${this.endpointDatasetAPI}`)
  }

  uploadModel(file:File){
    const formData = new FormData();
    formData.append('file', file);  
    return this.http.post(`${this.endpointModelAPI}`, formData)
  }

  listModels(){
    return this.http.get(`${this.endpointModelAPI}`)
  }

}

export interface FileList {
  files: string[];
}
