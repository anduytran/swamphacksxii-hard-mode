// A native modal <dialog> (focus trap, Esc to close) driven by an `open` prop. Closes from any
// [data-close] button or a tap on the backdrop.
import { useEffect, useRef, type ReactNode } from "react";

interface Props {
  open: boolean;
  onClose(): void;
  className: string;
  labelledBy: string;
  children: ReactNode;
}

export function Dialog({ open, onClose, className, labelledBy, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      className={className}
      aria-labelledby={labelledBy}
      onClose={onClose}
      onClick={(e) => {
        const target = e.target as HTMLElement;
        if (target === ref.current || target.closest("[data-close]")) onClose();
      }}
    >
      <div className="dialog-inner">{children}</div>
    </dialog>
  );
}
