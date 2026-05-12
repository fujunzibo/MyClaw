/**
 * MyClaw wordmark: use projectName in the nav; this is the paw mark only.
 */
interface CatPawIconProps {
  size: number;
  className?: string;
}

const LOGO_SRC = "/brand-mark.svg";

export function CatPawIcon({ size, className = "" }: CatPawIconProps) {
  return (
    <img
      src={LOGO_SRC}
      alt=""
      width={size}
      height={size}
      className={className}
      style={{
        display: "block",
        margin: "0 auto",
        objectFit: "contain",
      }}
      aria-hidden
    />
  );
}
