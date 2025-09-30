import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassificationOutputComponent } from './classification-output.component';

describe('ClassificationOutputComponent', () => {
  let component: ClassificationOutputComponent;
  let fixture: ComponentFixture<ClassificationOutputComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ClassificationOutputComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ClassificationOutputComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
