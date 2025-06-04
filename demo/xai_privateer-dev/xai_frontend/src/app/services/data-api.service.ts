import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class DataApiService {

  readonly endpointDataAPI = "http://127.0.0.1:5000/api/datasets"
  readonly endpointModelAPI = "http://127.0.0.1:5001/api/models"

  constructor(private http:HttpClient) { }

  uploadDataset(file:File){
    const formData = new FormData();
    formData.append('file', file);  
    return this.http.post(`${this.endpointDataAPI}/${file.name}`, formData)
  }

  listDatasets(){
    return this.http.get<FileList>(`${this.endpointDataAPI}/list`)
  }

  uploadModel(file:File){
    const formData = new FormData();
    formData.append('file', file);  
    return this.http.post(`${this.endpointModelAPI}/${file.name}`, formData)
  }

  listModels(){
    return this.http.get<FileList>(`${this.endpointModelAPI}/list`)
  }

}

export interface FileList {
  files: string[];
}
