import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Shap1Component } from './shap-1.component';

describe('Shap1Component', () => {
  let component: Shap1Component;
  let fixture: ComponentFixture<Shap1Component>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Shap1Component]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Shap1Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
