import { ReactNode } from "react";
import { clsx } from "clsx";

interface CardProps {
  children: ReactNode;
  className?: string;
}

export default function Card({ children, className }: CardProps) {
  return (
    <div
      className={clsx(
        "bg-white dark:bg-gray-900",
        "rounded-xl shadow-lg",
        "border border-gray-200 dark:border-gray-800",
        className
      )}
    >
      {children}
    </div>
  );
}
