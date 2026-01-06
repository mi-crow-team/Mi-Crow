import { ReactNode } from "react";

type RunHistorySidebarProps<T> = {
  title: string;
  items: T[];
  emptyMessage: string;
  renderItem: (item: T, index: number) => ReactNode;
  onDelete?: (item: T, index: number) => void;
  getItemKey?: (item: T, index: number) => string | number;
};

export function RunHistorySidebar<T>({ 
  title, 
  items, 
  emptyMessage, 
  renderItem,
  onDelete,
  getItemKey
}: RunHistorySidebarProps<T>) {
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
          {items.map((item, idx) => {
            const key = getItemKey ? getItemKey(item, idx) : idx;
            return (
              <div key={key} className="rounded-md border border-slate-200/60 bg-white/80 p-2 shadow-sm group">
                <div className="flex items-start gap-2">
                  <div className="flex-1 min-w-0">
              {renderItem(item, idx)}
            </div>
                  {onDelete && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(item, idx);
                      }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-red-600 hover:text-red-800 text-xs px-2 py-1 rounded hover:bg-red-50"
                      title="Delete"
                    >
                      ✕
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


