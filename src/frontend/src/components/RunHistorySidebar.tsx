import { ReactNode } from "react";

type RunHistorySidebarProps<T> = {
  title: string;
  items: T[];
  emptyMessage: string;
  renderItem: (item: T, index: number) => ReactNode;
};

export function RunHistorySidebar<T>({ title, items, emptyMessage, renderItem }: RunHistorySidebarProps<T>) {
  return (
    <div className="space-y-2 text-sm h-full">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-mi_crow-600">History</div>
        <div className="text-xs text-slate-500">{title}</div>
      </div>
      {items.length === 0 ? (
        <p className="text-xs text-slate-500">{emptyMessage}</p>
      ) : (
        <div className="space-y-2 pr-1 h-full">
          {items.map((item, idx) => (
            <div key={idx} className="rounded-md border border-slate-200/60 bg-white/80 p-2 shadow-sm">
              {renderItem(item, idx)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


