import { Component } from '@angular/core';
import { ShapApiService } from '../../services/shap-api.service';
import { HeatmapComponent } from '../../general-components/heatmap/heatmap.component';

@Component({
  selector: 'app-classification-output',
  imports: [HeatmapComponent],
  templateUrl: './classification-output.component.html',
  styleUrl: './classification-output.component.css'
})
export class ClassificationOutputComponent {

  shapValues: number[][]
  data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

  constructor(private shap: ShapApiService) {
    console.log(JSON.stringify(this.shap.shapReport.model_output))
    const input = this.convertTo2DArray(this.shap.shapReport.shap_values) as number[][]
    this.shapValues = this.subtractMatrices(input, this.shap.shapReport.model_output) as number[][]
    this.data = this.shapValues
    this.columnLabels = this.shap.labels
  }

  convertTo2DArray(obj: any, groupSize = 8) {
    // Get all values from the object in order
    const values = Object.values(obj);

    // Create 2D array by grouping consecutive values
    const result = [];

    for (let i = 0; i < values.length; i += groupSize) {
      // Extract a slice of 'groupSize' elements
      const group = values.slice(i, i + groupSize);
      result.push(group);
    }

    return result;
  }

  subtractMatrices(mat1: number[][], mat2: number[][]): number[][] | null {
    // Check if matrices have the same dimensions
    if (!mat1.length || !mat2.length || mat1.length !== mat2.length || mat1[0].length !== mat2[0].length) {
      console.error("Matrices must have the same dimensions");
      return null;
    }

    // Initialize result matrix with same dimensions
    const rows = mat1.length;
    const cols = mat1[0].length;
    const result: number[][] = [];

    // Perform element-wise subtraction
    for (let i = 0; i < rows; i++) {
      result[i] = [];
      for (let j = 0; j < cols; j++) {
        result[i][j] = mat1[i][j] - mat2[i][j];
      }
    }

    return result;
  }
}
