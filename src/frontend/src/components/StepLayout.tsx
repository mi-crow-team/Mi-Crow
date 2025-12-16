import { ReactNode } from "react";

import { Card, SectionTitle } from "@/components/ui";

type StepLayoutProps = {
  title: string;
  description?: string;
  steps: ReactNode;
  sidebar?: ReactNode;
};

export function StepLayout({ title, description, steps, sidebar }: StepLayoutProps) {
  return (
    <div className="space-y-4">
      <div>
        <SectionTitle>{title}</SectionTitle>
        {description && <p className="text-sm text-slate-500 mt-1">{description}</p>}
      </div>
      <div className="grid gap-2 lg:grid-cols-[minmax(260px,0.9fr)_minmax(0,1.6fr)] items-start">
        {sidebar && (
          <Card className="h-full space-y-2">{sidebar}</Card>
        )}
        <Card className="space-y-4">{steps}</Card>
      </div>
    </div>
  );
}

type StepCardProps = {
  step: number;
  title: string;
  description?: string;
  children: ReactNode;
};

export function StepCard({ step, title, description, children }: StepCardProps) {
  return (
    <Card className="space-y-3">
      <div className="space-y-1">
        <div className="text-xs font-semibold uppercase tracking-wide text-amber-600">
          Step {step} - {title}
        </div>
        {description && <p className="text-xs text-slate-500">{description}</p>}
      </div>
      {children}
    </Card>
  );
}


