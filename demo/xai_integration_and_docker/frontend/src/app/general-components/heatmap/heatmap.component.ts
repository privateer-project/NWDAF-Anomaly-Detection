import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';


@Component({
  selector: 'app-heatmap',
  imports: [CommonModule],
  templateUrl: './heatmap.component.html',
  styleUrl: './heatmap.component.css'
})
export class HeatmapComponent implements OnChanges {

  @Input() data: number[][] = [];
  @Input() colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  @Input() threshold: number = 50;
  @Input() columnLabels: string[] = [];

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data']) {
      console.log('Heatmap data changed:', this.data);
      console.log('Heatmap columnLabels:', this.columnLabels);
      console.log('Heatmap data dimensions:', this.data?.length, 'x', this.data?.[0]?.length);
      console.log('Heatmap data sample:', this.data?.[0]?.slice(0, 3));
      console.log('Heatmap data flat sample:', this.data?.flat()?.slice(0, 5));
    }
  }

  get rows(): number {
    return this.data.length;
  }

  get columns(): number {
    return this.data[0]?.length || 0;
  }

  getColor(value: number): string {
    if (!this.data || this.data.length === 0 || this.data[0].length === 0) {
      console.log('getColor: no data, returning first color');
      return this.colors[0]; // Return first color if no data
    }
    
    const flatData = this.data.flat();
    if (flatData.length === 0) {
      console.log('getColor: flatData empty, returning first color');
      return this.colors[0];
    }
    
    const max = Math.max(...flatData);
    const min = Math.min(...flatData);
    const range = max - min;

    console.log(`getColor: value=${value}, min=${min}, max=${max}, range=${range}`);

    if (range === 0) {
      console.log('getColor: range is 0, returning first color');
      return this.colors[0]; // All values are the same
    }

    const index = Math.floor(((value - min) / range) * (this.colors.length - 1));
    const finalIndex = Math.max(0, Math.min(index, this.colors.length - 1));
    console.log(`getColor: index=${index}, finalIndex=${finalIndex}, color=${this.colors[finalIndex]}`);
    return this.colors[finalIndex];
  }

}
