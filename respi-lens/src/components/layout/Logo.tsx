import { Link } from 'react-router-dom'
import respiLogo from '../../assets/respi_logo.png'

export function Logo() {
    return (
        <Link
            to="/"
            className="flex items-center gap-2.5 group focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)] rounded"
        >
            <img
                src={respiLogo}
                alt="Respi"
                className="h-10 w-10 object-contain transition-transform duration-150 group-hover:scale-[1.04] shrink-0"
            />
            <div className="flex flex-col leading-none">
                <span className="text-[15px] font-bold tracking-tight text-[var(--fg)]">
                    Respi
                </span>
                <span className="text-[10px] font-semibold text-[var(--fg-muted)] uppercase tracking-[0.14em] mt-0.5">
                    Clinical
                </span>
            </div>
        </Link>
    )
}
