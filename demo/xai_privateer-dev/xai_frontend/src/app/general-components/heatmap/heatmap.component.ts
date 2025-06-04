import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';


@Component({
  selector: 'app-heatmap',
  imports: [CommonModule],
  templateUrl: './heatmap.component.html',
  styleUrl: './heatmap.component.css'
})
export class HeatmapComponent {

  @Input() data: number[][] = [];
  @Input() colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  @Input() threshold: number = 50;
  @Input() columnLabels: string[] = [];

  get rows(): number {
    return this.data.length;
  }

  get columns(): number {
    return this.data[0]?.length || 0;
  }

  getColor(value: number): string {
    const max = Math.max(...this.data.flat());
    const min = Math.min(...this.data.flat());
    const range = max - min;

    const index = Math.floor(((value - min) / range) * (this.colors.length - 1));
    return this.colors[index];
  }

}
