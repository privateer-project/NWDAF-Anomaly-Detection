import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Shap8Component } from './shap-8.component';

describe('Shap8Component', () => {
  let component: Shap8Component;
  let fixture: ComponentFixture<Shap8Component>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Shap8Component]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Shap8Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
