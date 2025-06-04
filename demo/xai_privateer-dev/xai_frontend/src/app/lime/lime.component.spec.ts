import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LimeComponent } from './lime.component';

describe('LimeComponent', () => {
  let component: LimeComponent;
  let fixture: ComponentFixture<LimeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LimeComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LimeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
