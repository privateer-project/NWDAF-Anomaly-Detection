export enum RequestState {
  Not_Initiated = 'Not_Initiated',
  Initiated = 'Initiated',
  Completed = "Completed",
  Error = "Error"
}

export interface FileList {
  files: string[];
}