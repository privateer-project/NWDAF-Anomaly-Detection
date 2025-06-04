import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProductionSettingsComponent } from './production-settings.component';

describe('ProductionSettingsComponent', () => {
  let component: ProductionSettingsComponent;
  let fixture: ComponentFixture<ProductionSettingsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProductionSettingsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ProductionSettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
